//  Created by Aidan Gomez on 2015-08-13.
//  Copyright (c) 2015 Venture Media. All rights reserved.


#pragma once
#include <valarray>
#include <memory>

#include "caffe/solver.hpp"
#include "caffe/util/fft.hpp"
#include "caffe/util/read_audio.hpp"


namespace caffe {
    
template <typename T>
class KernelGen {
public:
    KernelGen(int numKernels, int kernelSize) : kNumKernels(numKernels), kKernelSize(kernelSize){}
    KernelGen(KernelGen& other) : kNumKernels(other.kNumKernels), kKernelSize(other.kKernelSize){}
    virtual ~KernelGen(){}
    virtual KernelGen& operator=(const KernelGen& rhs) {
        kNumKernels = rhs.kNumKernels;
        kKernelSize = rhs.kKernelSize;
        return *this;
    }
    virtual inline void generateKernel(T* data) {
        auto shape = {kNumKernels, 1, 1, kKernelSize};
        auto blob = std::make_shared<caffe::Blob<T>>(shape);
        
        auto kernel = kernelGen();
        std::copy(std::begin(kernel), std::begin(kernel) + (kNumKernels * kKernelSize), data);
    }
    static inline T curve(T x, T a, T b) { return std::exp((x-a)*(x-a)/b); }
    
protected:
    virtual std::valarray<T> kernelGen() = 0;
    
protected:
    int kNumKernels;
    int kKernelSize;
};

template <typename T>
class ThickKernel : public KernelGen<T> {
public:
    ThickKernel(int numKernels, int kernelSize) : KernelGen<T>(numKernels, kernelSize) {}
    ThickKernel(ThickKernel& other) : KernelGen<T>(other) {}
    ~ThickKernel(){}
    ThickKernel operator=(ThickKernel rhs) {
        KernelGen<T>::operator=(rhs);
        return *this;
    }
    
protected:
    inline std::valarray<T> kernelGen() {
        const auto numKernels = KernelGen<T>::kNumKernels;
        const auto kernelSize = KernelGen<T>::kKernelSize;
        
        auto step = static_cast<double>(kernelSize) / numKernels;
        auto kernelWeights = std::valarray<T>(kernelSize * numKernels);
        
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
class ThinKernel : public KernelGen<T> {
public:
    ThinKernel(int numKernels, int kernelSize) : KernelGen<T>(numKernels, kernelSize) {}
    ThinKernel(ThinKernel& other) : KernelGen<T>(other) {}
    ~ThinKernel(){}
    
protected:
    inline std::valarray<T> kernelGen() {
        const auto numKernels = KernelGen<T>::kNumKernels;
        const auto kernelSize = KernelGen<T>::kKernelSize;
        
        auto step = static_cast<double>(kernelSize) / numKernels;
        auto kernelWeights = std::valarray<T>(kernelSize * numKernels);
        
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
    double kB = -0.0005;
};

template <typename T>
class FFTKernel : public KernelGen<T> {
public:
    FFTKernel(int numKernels, int kernelSize) : KernelGen<T>(numKernels, kernelSize) {}
    FFTKernel(FFTKernel& other) : KernelGen<T>(other) {}
    ~FFTKernel(){}
    
protected:
    inline std::valarray<T> kernelGen() {
        const auto numKernels = KernelGen<T>::kNumKernels;
        const auto kernelSize = KernelGen<T>::kKernelSize;
        
        auto data = std::valarray<T>(numKernels * kernelSize);
        auto fftBuffer = std::valarray<T>(2 * kernelSize);
        
        const auto options = caffe::FFTOptions{};
        
        for (auto i = 0; i < numKernels; ++i) {
            if (i < 84) {
                const auto filename = "data/Training/Notes/AcousticGrandPiano_YDP/" + std::to_string(i + 24) + ".caf";
                caffe::ReadAudioFile(filename, std::begin(fftBuffer), 2 * kernelSize);
                
                auto fft = caffe::FastFourierTransform<T>(2 * kernelSize, options);
                fft.process(std::begin(fftBuffer), 2 * kernelSize);
                                
                 std::move(std::begin(fftBuffer), std::begin(fftBuffer) + kernelSize, std::begin(data) + (i * kernelSize));
            } else {
                auto dataStart = std::begin(data) + ((i % 84) * kernelSize);
                std::move_backward(dataStart, dataStart + kernelSize, std::begin(data) + (i * kernelSize));
            }
        }
        
        return data;
    }
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
    inline void assignConvolutionWeights(std::shared_ptr<caffe::Net<T>> net, caffe::FillerParameter_PeakType kernelType) {
    for (auto layer : net->layers()) {
        if (layer->type() == "Convolution") {
            auto parameters = layer->layer_param().convolution_param();
            auto numKernels = parameters.num_output();
            auto kernelSize = parameters.kernel_w() * parameters.kernel_h();
            
            std::shared_ptr<KernelGen<T>> kernel;
            switch (kernelType) {
                case caffe::FillerParameter_PeakType_FFT:
                    kernel = std::make_shared<FFTKernel<T>>(numKernels, kernelSize);
                    break;
                case caffe::FillerParameter_PeakType_THICK:
                    kernel = std::make_shared<ThickKernel<T>>(numKernels, kernelSize);
                    break;
                case caffe::FillerParameter_PeakType_THIN:
                    kernel = std::make_shared<ThinKernel<T>>(numKernels, kernelSize);
                    break;
            }
            
            auto data = static_cast<T*>(layer->blobs()[0]->data()->mutable_cpu_data()); // first blob is weights, second is biases
            kernel->generateKernel(data);
            break;
        }
    }
}
    
template <typename T>
inline void assignConvolutionWeights(T* data, int numKernels, int kernelSize, caffe::FillerParameter_PeakType kernelType) {
    std::shared_ptr<KernelGen<T>> kernel;
    switch (kernelType) {
        case caffe::FillerParameter_PeakType_FFT:
            kernel = std::make_shared<FFTKernel<T>>(numKernels, kernelSize);
            break;
        case caffe::FillerParameter_PeakType_THICK:
            kernel = std::make_shared<ThickKernel<T>>(numKernels, kernelSize);
            break;
        case caffe::FillerParameter_PeakType_THIN:
            kernel = std::make_shared<ThinKernel<T>>(numKernels, kernelSize);
            break;
    }
    kernel->generateKernel(data);
}

} // namespace caffe