//  Created by Aidan Gomez on 2015-08-13.
//  Copyright (c) 2015 Venture Media. All rights reserved.


#pragma once
#include <vector>
#include <memory>

#include "caffe/solver.hpp"



namespace helper {
    
template <typename T>
class KernelGen {
public:
    KernelGen(int numKernels, int kernelSize) : kNumKernels(numKernels), kKernelSize(kernelSize){}
    virtual inline void generateKernel(T* data) {
        auto shape = {kNumKernels, 1, 1, kKernelSize};
        auto blob = std::make_shared<caffe::Blob<T>>(shape);
        
        auto kernel = kernelGen();
        std::copy(std::begin(kernel), std::begin(kernel) + (kNumKernels * kKernelSize), data);
    }
    static inline T curve(T x, T a, T b) { return std::exp((x-a)*(x-a)/b); }
    
protected:
    virtual inline std::vector<T> kernelGen() { return {}; };
    
protected:
    const int kNumKernels;
    const int kKernelSize;
};

template <typename T>
class ThickKernel : public KernelGen<T> {
public:
    ThickKernel(int numKernels, int kernelSize) : KernelGen<T>(numKernels, kernelSize) {}
    
protected:
    inline std::vector<T> kernelGen() {
        const auto numKernels = KernelGen<T>::kNumKernels;
        const auto kernelSize = KernelGen<T>::kKernelSize;
        
        auto step = static_cast<double>(kernelSize) / numKernels;
        auto kernelWeights = std::vector<T>(kernelSize * numKernels);
        
        for (auto kernel = 0; kernel < numKernels; ++kernel) {
            auto peakLocation = step * kernel / kernelSize;
            for (auto position = 0; position < kernelSize; ++position) {
                auto x = static_cast<double>(position) / kernelSize;
                kernelWeights[(kernel * kernelSize) + position] = KernelGen<T>::curve(x, peakLocation, kB);
            }
        }
        
        return kernelWeights;
    }
    
protected:
    double kB = -0.005;
};
    
template <typename T>
std::shared_ptr<caffe::Blob<T>> generateKernel(int numKernels, int kernelSize);
template <typename T>
void assignConvolutionWeights(std::shared_ptr<caffe::Net<T>> net);

template <typename T>
inline void initializeSolverWeights(caffe::Solver<T> solver) {
    assignConvolutionWeights(solver.net());
    for (auto net : solver.test_nets()) {
        assignConvolutionWeights(net);
    }
}
    
template <typename T>
inline void assignConvolutionWeights(std::shared_ptr<caffe::Net<T>> net) {
    for (auto layer : net->layers()) {
        if (layer->type() == "Convolution") {
            auto parameters = layer->layer_param().convolution_param();
            auto numKernels = parameters.num_output();
            auto kernelSize = parameters.kernel_w() * parameters.kernel_h();
            
            auto kernel = ThickKernel<T>(numKernels, kernelSize);
            auto data = static_cast<T*>(layer->blobs()[0]->data()->mutable_cpu_data()); // first blob is weights, second is biases
            kernel.generateKernel(data);
            break;
        }
    }
}
    
template <typename T>
inline void assignConvolutionWeights(T* data, int numKernels, int kernelWidth) {
    auto kernel = ThickKernel<T>(numKernels, kernelWidth);
    kernel.generateKernel(data);
}

} // namespace helper