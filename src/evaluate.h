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

#ifndef EVALUATE_H_INCLUDED
#define EVALUATE_H_INCLUDED

#include <string>
#include <cstring>

#include "types.h"
#include "misc.h"
#include "iostream"

namespace Stockfish {

class Position;

namespace Eval {

constexpr inline int SmallNetThreshold = 1274, PsqtOnlyThreshold = 2389;

// The default net name MUST follow the format nn-[SHA256 first 12 digits].nnue
// for the build process (profile-build and fishtest) to work. Do not change the
// name of the macro or the location where this macro is defined, as it is used
// in the Makefile/Fishtest.
#define EvalFileDefaultNameBig "nn-ae6a388e4a1a.nnue"
#define EvalFileDefaultNameSmall "nn-baff1ede1f90.nnue"

namespace NNUE {
struct Networks;
struct AccumulatorCaches;
}

std::string trace(Position& pos, const Eval::NNUE::Networks& networks);

int   simple_eval(const Position& pos, Color c);
Value evaluate(const NNUE::Networks&          networks,
               const Position&                pos,
               Eval::NNUE::AccumulatorCaches& caches,
               int                            optimism);
}  // namespace Eval

// I have no idea how the nnue code works, thats why im doing it this way

class OutputWeightNet {
public:
    static const int INPUT_SIZE  = 5; 
    static const int HIDDEN_SIZE = 128;
    static const int OUTPUT_SIZE = 32;

    std::array<std::array<int8_t, HIDDEN_SIZE>, COLOR_NB> accumulator;
    OWNKey accInputKey = OWNKey();

    std::array<int8_t, INPUT_SIZE * HIDDEN_SIZE> l0Weights = {};
    std::array<int8_t, OUTPUT_SIZE * HIDDEN_SIZE * 2> l1Weights = {};
    std::array<int8_t, HIDDEN_SIZE> l0biases = {};
    std::array<int8_t, OUTPUT_SIZE> l1biases = {};

    static const int keyCombinations = 9 * 3 * 3 * 3 * 2;

    std::array<std::array<int8_t*, keyCombinations>, keyCombinations> cache;

    const std::array<int, PIECE_TYPE_NB> max = {0, 8, 2, 2, 2, 1};

    OutputWeightNet() {
        for (auto entry : cache)
          entry.fill(nullptr);

        accumulator[WHITE] = l0biases;
        accumulator[BLACK] = l0biases;
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
        int8_t out[OUTPUT_SIZE];

        memset(&out, 0, sizeof(int8_t) * OUTPUT_SIZE);

        for (int i = 0; i < OUTPUT_SIZE; i++) {
           for (int j = 0; j < HIDDEN_SIZE; j++) {
              int index = i * HIDDEN_SIZE + j;

              out[i] += relu(accumulator[WHITE][j]) * l1Weights[index];
              out[i] += relu(accumulator[BLACK][j]) * l1Weights[index + OUTPUT_SIZE * HIDDEN_SIZE];
           }

           entry[i] = out[i] + l1biases[i];
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

}  // namespace Stockfish

#endif  // #ifndef EVALUATE_H_INCLUDED
