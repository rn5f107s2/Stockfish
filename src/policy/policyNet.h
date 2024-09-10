#pragma once

#include "block.h"
#include "convLayer.h"

#include "../types.h"
#include "../bitboard.h"
#include "../search.h"
#include "../uci.h"

#include <array>
#include <cstdint>

using namespace Stockfish;

const int BLOCKS = 4;

int moveToLayer(int from, int to);
void bbsToPaddedInput(std::array<Bitboard, 13> &bitboards, bool stm, std::array<std::array<std::array<float, 10>, 10>, 12> &input);

struct Network {
    using Input  = std::array<std::array<std::array<float, 10>, 10>, 12>;
    using Output = std::array<std::array<std::array<float, 10>, 10>, 64>;

    ConvolutionalLayer<12, FILTER> cl1;
    std::array<Block, BLOCKS> blocks;
    ConvolutionalLayer<FILTER, 64> cl2;

    void scoreMoveList(Input &input, Search::RootMoves &ml, int(*policies)[64], bool stm) {
        Output &out = forward(input);

        float sum = 0.0f;
        float scores[218];

        for (size_t i = 0; i < ml.size(); i++) {
            Search::RootMove &rm = ml[i];
            int from     = int(rm.pv[0].from_sq()) ^ 7;
            int to       = int(rm.pv[0].to_sq())   ^ 7;

            if (!stm) {
                from ^= 56;
                to   ^= 56;
            }

            int layer    = moveToLayer(from, to);
            int fromFile = file_of(Square(from));
            int fromRank = rank_of(Square(from));

            sum += (scores[i] = std::exp(out[layer][fromRank + 1][fromFile + 1]));
        }

        for (size_t i = 0; i < ml.size(); i++)
            policies[ml[i].pv[0].from_sq()][ml[i].pv[0].to_sq()] = (scores[i] / sum) * 16384.0f;
    }

    void loadWeights(std::istream &in) {
        cl1.loadWeights(in);

        for (Block &b : blocks)
            b.loadWeights(in);

        cl2.loadWeights(in);
    }

    void loadDefault();

private:
    Output& forward(Input &input) {
        auto &out = cl1.forward(input);
              out = cl1.ReLUInplace();
        
        for (Block &b : blocks)
            out = b.forward(out);

        return cl2.forward(out);
    }
};