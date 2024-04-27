/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2024 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "evaluate.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <memory>

#include "nnue/network.h"
#include "nnue/nnue_misc.h"
#include "position.h"
#include "types.h"
#include "uci.h"
#include "nnue/nnue_accumulator.h"

namespace Stockfish {

// Returns a static, purely materialistic evaluation of the position from
// the point of view of the given color. It can be divided by PawnValue to get
// an approximation of the material advantage on the board in terms of pawns.
int Eval::simple_eval(const Position& pos, Color c) {
    return PawnValue * (pos.count<PAWN>(c) - pos.count<PAWN>(~c))
         + (pos.non_pawn_material(c) - pos.non_pawn_material(~c));
}

const int INPUT_SIZE  = 5; 
const int HIDDEN_SIZE = 64;
const int OUTPUT_SIZE = 32;

int l0Weights[INPUT_SIZE * HIDDEN_SIZE] = {};
int l1Weights[OUTPUT_SIZE * HIDDEN_SIZE] = {};
int l0biases [HIDDEN_SIZE] = {};
int l1biases [OUTPUT_SIZE] = {};

TUNE(SetRange(-7, 7), l0Weights, l0biases);
TUNE(SetRange(-4, 4), l1Weights);
TUNE(SetRange(-128, 127), l1biases);

class OutputWeightNet {
public:
    int16_t accumulator[COLOR_NB][HIDDEN_SIZE];
    OWNKey accInputKey = OWNKey();

    static const int keyCombinations = 9 * 3 * 3 * 3 * 2;

    std::array<std::array<int8_t*, keyCombinations>, keyCombinations> cache;

    const std::array<int, PIECE_TYPE_NB> max = {0, 8, 2, 2, 2, 1};

    OutputWeightNet() {
        for (auto entry : cache)
          entry.fill(nullptr);

        for (int i = 0; i < HIDDEN_SIZE; i++)
          accumulator[WHITE][i] = accumulator[BLACK][i] = l0biases[i];
    }

    ~OutputWeightNet() {
        for (auto entry : cache)
          for (auto* entry2 : entry)
            if (entry2)
              free(entry2);
    }

    int8_t* getWeights(const OWNKey &key, Color stm) {
        int8_t** entry = &cache[key.key(stm)][key.key(~stm)];

        if (*entry)
          return *entry;

        *entry = static_cast<int8_t*>(malloc(sizeof(int8_t) * OUTPUT_SIZE));

        updateAccumulator(key);
        forward(*entry);

        return *entry;
    }

    void forward(int8_t* entry)  {
        int16_t out[OUTPUT_SIZE];

        memset(&out, 0, sizeof(int8_t) * OUTPUT_SIZE);

        for (int i = 0; i < OUTPUT_SIZE; i++) {
           for (int j = 0; j < HIDDEN_SIZE; j++) {
              int index = i * HIDDEN_SIZE + j;

              out[i] += relu(accumulator[WHITE][j]) * l1Weights[index];
              out[i] += relu(accumulator[BLACK][j]) * l1Weights[index + OUTPUT_SIZE * HIDDEN_SIZE];
           }

           entry[i] = int8_t((out[i] + l1biases[i]) / 256);
        }
    }

    int8_t relu(int8_t val) {
        return std::max(val, int8_t(0));
    }

    void updateAccumulator(const OWNKey &newKey) {
        for (Color c : {WHITE, BLACK}) {
            for (PieceType pt = PAWN; pt <= QUEEN; ++pt) {
                int diff = newKey.count(c, pt) - accInputKey.count(c, pt);

                updateAccumulator(c, pt - 1, diff);
                accInputKey.pieceCount[c][pt] += diff;
            }
        }
    }

    void updateAccumulator(Color c, int index, int difference) {
        if (!difference)
          return;

        for (int i = 0; i < HIDDEN_SIZE; i++)
            accumulator[c][i] += difference * l0Weights[index * HIDDEN_SIZE + i];
    }
};

OutputWeightNet* own = new OutputWeightNet;

// Evaluate is the evaluator for the outer world. It returns a static evaluation
// of the position from the point of view of the side to move.
Value Eval::evaluate(const Eval::NNUE::Networks&    networks,
                     const Position&                pos,
                     Eval::NNUE::AccumulatorCaches& caches,
                     int                            optimism) {

    assert(!pos.checkers());

    int  simpleEval = simple_eval(pos, pos.side_to_move());
    bool smallNet   = std::abs(simpleEval) > SmallNetThreshold;
    bool psqtOnly   = std::abs(simpleEval) > PsqtOnlyThreshold;
    int  nnueComplexity;
    int  v;

    Value nnue = smallNet ? networks.small.evaluate(pos, nullptr, true, &nnueComplexity, psqtOnly)
                          : networks.big.evaluate(pos, &caches.big, true, &nnueComplexity, false, own->getWeights(pos.ownKey, pos.side_to_move()));

    const auto adjustEval = [&](int optDiv, int nnueDiv, int npmDiv, int pawnCountConstant,
                                int pawnCountMul, int npmConstant, int evalDiv,
                                int shufflingConstant, int shufflingDiv) {
        // Blend optimism and eval with nnue complexity and material imbalance
        optimism += optimism * (nnueComplexity + std::abs(simpleEval - nnue)) / optDiv;
        nnue -= nnue * (nnueComplexity * 5 / 3) / nnueDiv;

        int npm = pos.non_pawn_material() / npmDiv;
        v       = (nnue * (npm + pawnCountConstant + pawnCountMul * pos.count<PAWN>())
             + optimism * (npmConstant + npm))
          / evalDiv;

        // Damp down the evaluation linearly when shuffling
        int shuffling = pos.rule50_count();
        v             = v * (shufflingConstant - shuffling) / shufflingDiv;
    };

    if (!smallNet)
        adjustEval(524, 32395, 66, 942, 11, 139, 1058, 178, 204);
    else if (psqtOnly)
        adjustEval(517, 32857, 65, 908, 7, 155, 1006, 224, 238);
    else
        adjustEval(515, 32793, 63, 944, 9, 140, 1067, 206, 206);

    // Guarantee evaluation does not hit the tablebase range
    v = std::clamp(v, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);

    return v;
}

// Like evaluate(), but instead of returning a value, it returns
// a string (suitable for outputting to stdout) that contains the detailed
// descriptions and values of each evaluation term. Useful for debugging.
// Trace scores are from white's point of view
std::string Eval::trace(Position& pos, const Eval::NNUE::Networks& networks) {

    auto caches = std::make_unique<Eval::NNUE::AccumulatorCaches>();

    if (pos.checkers())
        return "Final evaluation: none (in check)";

    std::stringstream ss;
    ss << std::showpoint << std::noshowpos << std::fixed << std::setprecision(2);
    ss << '\n' << NNUE::trace(pos, networks, *caches) << '\n';

    ss << std::showpoint << std::showpos << std::fixed << std::setprecision(2) << std::setw(15);

    Value v = networks.big.evaluate(pos, &caches->big, false);
    v       = pos.side_to_move() == WHITE ? v : -v;
    ss << "NNUE evaluation        " << 0.01 * UCIEngine::to_cp(v, pos) << " (white side)\n";

    v = evaluate(networks, pos, *caches, VALUE_ZERO);
    v = pos.side_to_move() == WHITE ? v : -v;
    ss << "Final evaluation       " << 0.01 * UCIEngine::to_cp(v, pos) << " (white side)";
    ss << " [with scaled NNUE, ...]";
    ss << "\n";

    return ss.str();
}

}  // namespace Stockfish
