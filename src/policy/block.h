#pragma once

#include "convLayer.h"
#include "batchnorm.h"

const int FILTER = 64;

struct Block {
    using Input  = std::array<std::array<std::array<float, 10>, 10>, FILTER>;
    using Output = Input;

    ConvolutionalLayer<FILTER, FILTER> cl1;
    Batchnorm<FILTER>                  bn1;
    ConvolutionalLayer<FILTER, FILTER> cl2;
    Batchnorm<FILTER>                  bn2;

    Output& forward(Input& input) {
        Input identity;
        memcpy(&identity, &input, sizeof(Input));

        Output& out = cl1.forward(input);
        bn1.forward(out);
        out = bn1.ReLUInplace();
        out = cl2.forward(out);
        bn2.forward(out, &identity);
        out = bn2.ReLUInplace();

        return out;
    }

    void loadWeights(std::istream& in) {
        cl1.loadWeights(in);
        bn1.loadWeights(in);
        cl2.loadWeights(in);
        bn2.loadWeights(in);
    }
};