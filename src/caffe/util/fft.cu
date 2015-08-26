//  Created by Aidan Gomez on 2015-08-26.
//  Copyright (c) 2015 Venture Media. All rights reserved.

#include "caffe/common.hpp"
#include "caffe/util/fft.hpp"

#include <cufft.h>

namespace caffe {

struct FastFourierTransformPImpl {
    cufftHandle plan;
};

template <>
FastFourierTransform_gpu<double>::FastFourierTransform_gpu(int packetSize, FFTOptions options)
: _log2Size(std::ceil(std::log2(packetSize))),
_packetSize(static_cast<int>(std::exp2(_log2Size))),
_pimpl(new FastFourierTransformPImpl()),
_options(options)
{}

template <>
FastFourierTransform_gpu<float>::FastFourierTransform_gpu(int packetSize, FFTOptions options)
: _log2Size(std::ceil(std::log2(packetSize))),
_packetSize(static_cast<int>(std::exp2(_log2Size))),
_pimpl(new FastFourierTransformPImpl()),
_options(options)
{}

template <typename Dtype>
FastFourierTransform_gpu<Dtype>::~FastFourierTransform_gpu() {}

template <>
int FastFourierTransform_gpu<double>::process(double* data, int size) {
    CHECK_EQ(size, _packetSize);

    CHECK_EQ(cufftPlan1d(&(_pimpl->plan), size, CUFFT_D2Z, 1), CUFFT_SUCCESS) << "Creation of plan failed.";
    CHECK_EQ(cufftExecD2Z(_pimpl->plan, reinterpret_cast<cufftDoubleReal*>(data), reinterpret_cast<cufftDoubleComplex*>(data)), CUFFT_SUCCESS) << "Execution of cuFFT failed.";
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess) << "CUDA failed to synchronize.";
    CHECK_EQ(cufftDestroy(_pimpl->plan), CUFFT_SUCCESS) << "Failed to destroy cuFFT.";

    return size;
}

template <>
int FastFourierTransform_gpu<float>::process(float* data, int size) {
  CHECK_EQ(size, _packetSize);

  CHECK_EQ(cufftPlan1d(&(_pimpl->plan), size, CUFFT_R2C, 1), CUFFT_SUCCESS) << "Creation of plan failed.";
  CHECK_EQ(cufftExecR2C(_pimpl->plan, reinterpret_cast<cufftReal*>(data), reinterpret_cast<cufftComplex*>(data)), CUFFT_SUCCESS) << "Execution of cuFFT failed.";
  CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess) << "CUDA failed to synchronize.";
  CHECK_EQ(cufftDestroy(_pimpl->plan), CUFFT_SUCCESS) << "Failed to destroy cuFFT.";

  return size;
}

INSTANTIATE_CLASS(FastFourierTransform_gpu);

} // namespace caffe
