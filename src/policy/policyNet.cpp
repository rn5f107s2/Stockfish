#include "policyNet.h"

#include "../incbin/incbin.h"
#include "../bitboard.h"
#include "../evaluate.h"

INCBIN(policy, PolicyFileDefaultName);

void Network::loadDefault() {
    size_t size = 1365504;
    std::string w(reinterpret_cast<const char*>(gpolicyData), size);
    std::stringstream weights(w);
    loadWeights(weights);
}

int moveToLayer(int from, int to) {
    const int V_POS_OFFSET   = -1;                //  0  1  2  3  4  5  6
    const int V_NEG_OFFSET   = V_POS_OFFSET  + 7; //  7  8  9 10 11 12 13
    const int H_POS_OFFSET   = V_NEG_OFFSET  + 7; // 14 15 16 17 18 19 20
    const int H_NEG_OFFSET   = H_POS_OFFSET  + 7; // 21 22 23 24 25 26 27
    const int D1_POS_OFFSET  = H_NEG_OFFSET  + 7; // 28 29 30 31 32 33 34 
    const int D1_NEG_OFFSET  = D1_POS_OFFSET + 7; // 35 36 37 38 39 40 41
    const int D2_POS_OFFSET  = D1_NEG_OFFSET + 7; // 42 43 44 45 46 47 48 
    const int D2_NEG_OFFSET  = D2_POS_OFFSET + 7; // 49 50 51 52 53 54 55
    const int KNIGHT_OFFSET1 = 56;
    const int KNIGHT_OFFSET2 = 58;
    const int KNIGHT_OFFSET3 = 60;
    const int KNIGHT_OFFSET4 = 62;

    int fromFile =  from & 0b000111;
    int fromRank = (from & 0b111000) >> 3;
    int toFile   =    to & 0b000111;
    int toRank   = (  to & 0b111000) >> 3;

    if (fromFile == toFile)
        return fromRank > toRank ? (fromRank - toRank + V_POS_OFFSET) : (toRank - fromRank + V_NEG_OFFSET);

    if (fromRank == toRank)
        return fromFile > toFile ? (fromFile - toFile + H_POS_OFFSET) : (toFile - fromFile + H_NEG_OFFSET);

    if (fromRank + fromFile == toRank + toFile)
        return fromFile > toFile ? (fromFile - toFile + D1_POS_OFFSET) : (toFile - fromFile + D1_NEG_OFFSET);

    if (fromRank - fromFile == toRank - toFile)
        return fromFile > toFile ? (fromFile - toFile + D2_POS_OFFSET) : (toFile - fromFile + D2_NEG_OFFSET);

    if (fromRank - toRank   > 1)
        return KNIGHT_OFFSET1 + (fromFile > toFile);

    if (toRank   - fromRank > 1)
        return KNIGHT_OFFSET2 + (fromFile > toFile);

    if (toFile   - fromFile > 1)
        return KNIGHT_OFFSET3 + (fromRank > toRank);

    if (fromFile - toFile   > 1)
        return KNIGHT_OFFSET4 + (fromRank > toRank);

    return 0;
}

void bbsToPaddedInput(std::array<Bitboard, 13> &bitboards, bool stm, std::array<std::array<std::array<float, 10>, 10>, 12> &input) {
    memset(&input, 0, sizeof(input));

    for (int pc = 0; pc < 13; pc++) {
        Bitboard pieceBB = bitboards[pc];

        while (pieceBB) {
            int square = int(pop_lsb(pieceBB)) ^ 7;
            int piece  = pc;

            if (!stm) {
                square ^= 56;
                piece = piece - (piece > 5 ? 6 : -6);
            }

            int rank = square / 8;
            int file = square % 8;

            input[piece][rank + 1][file + 1] = 1.0f;
        }
    }
}