//  Created by Aidan Gomez on 2015-08-13.
//  Copyright (c) 2015 Venture Media. All rights reserved.


#pragma once
#include <vector>
#include <memory>

#include "caffe/solver.hpp"



namespace caffe {
    
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
class ThinKernel : public KernelGen<T> {
public:
    ThinKernel(int numKernels, int kernelSize) : KernelGen<T>(numKernels, kernelSize) {}
    ThinKernel(ThinKernel& other) : KernelGen<T>(other) {}
    ~ThinKernel(){}

protected:
    inline std::valarray<T> kernelGen() {
        const auto numKernels = KernelGen<T>::kNumKernels;
        const auto kernelSize = KernelGen<T>::kKernelSize;

        auto kernelWeights = std::valarray<T>(kernelSize * numKernels);

        T scale = 0.1;
        caffe_rng_uniform<T>(numKernels * kernelSize, -scale, scale,
                             std::begin(kernelWeights));

        std::normal_distribution<> random_distribution(0, 0.5);
        std::function<T()> variate_generator = std::bind(random_distribution, std::ref(*caffe_rng()));

        for (auto n = 0; n < numKernels; n += 1) {
            auto ng = 2 * (variate_generator() + 0.5);
            for (auto g = 0; g < ng; g += 1) {
                auto height = variate_generator();
                auto mid = kernelSize/2 + variate_generator()*kernelSize;
                auto width = 1 + variate_generator();
                for (auto i = 0; i < kernelSize; i += 1) {
                    auto offset = n * kernelSize + i;
                    kernelWeights[offset] += gaussian(i, height, mid, width);
                }
            }
        }

        return kernelWeights;
    }
    inline T gaussian(T x, T height, T mid, T width) {
        x -= mid;
        return height * std::exp(-x*x / (2*width*width));
    }
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
                const auto filename = "data/Training/Notes/AcousticGrandPiano_YDP/" + std::to_string(i + 24) + ".aiff";
                caffe::ReadAudioFile(filename, std::begin(fftBuffer), 2 * kernelSize);

                caffe::FastFourierTransform_cpu<T> fft(2 * kernelSize, options);
                fft.process(std::begin(fftBuffer), 2 * kernelSize);

                std::transform(std::begin(fftBuffer), std::begin(fftBuffer) + 2 * kernelSize, std::begin(fftBuffer), [kernelSize](const T& a){
                    return  a / (2 * kernelSize);
                });

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
class WaveletKernel : public KernelGen<T> {
public:
    WaveletKernel(int numKernels, int kernelSize) : KernelGen<T>(numKernels, kernelSize) {}
    WaveletKernel(WaveletKernel& other) : KernelGen<T>(other) {}
    ~WaveletKernel(){}

protected:
    inline std::valarray<T> kernelGen() {
        const auto numKernels = KernelGen<T>::kNumKernels;
        const auto kernelSize = KernelGen<T>::kKernelSize;

        auto data = std::valarray<T>(numKernels * kernelSize);

        T scale = 0.1;
        caffe_rng_uniform<T>(numKernels * kernelSize, -scale, scale,
                             std::begin(data));

        std::uniform_real_distribution<> fdistribution(20, 20000);
        std::function<T()> fgenerator = std::bind(fdistribution, std::ref(*caffe_rng()));

        std::uniform_real_distribution<> pdistribution(-M_PI/2, M_PI/2);
        std::function<T()> pgenerator = std::bind(pdistribution, std::ref(*caffe_rng()));

        const auto width = kernelSize;
        for (auto n = 0; n < numKernels; n += 1) {
            const auto f = fgenerator();
            const auto p = pgenerator();
            const auto offset = n * kernelSize;
            generate(f, p, std::begin(data) + offset, width);
        }

        return data;
    }
    inline void generate(T frequency, T phase, T* data, int capacity) {
        const auto sampleRate = 44100.0;
        const auto dt = 1.0 / sampleRate;
        auto time = 0.0;
        for (std::size_t i = 0; i < capacity; i += 1) {
            data[i] = std::sin(2 * M_PI * frequency * time + phase);
            time += dt;
        }
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
inline void assignConvolutionWeights(std::shared_ptr<caffe::Net<T>> net) {
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
                case caffe::FillerParameter_PeakType_WAVELET:
                    kernel = std::make_shared<WaveletKernel<T>>(numKernels, kernelSize);
                    break;
            }

            auto data = static_cast<T*>(layer->blobs()[0]->data()->mutable_cpu_data()); // first blob is weights, second is biases
            kernel.generateKernel(data);
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
        case caffe::FillerParameter_PeakType_WAVELET:
            kernel = std::make_shared<WaveletKernel<T>>(numKernels, kernelSize);
            break;
    }
    kernel->generateKernel(data);
}

} // namespace caffe
