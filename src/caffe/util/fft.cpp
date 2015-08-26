
#include <valarray>
#include <cmath>

#include <aquila/transform/AquilaFft.h>
#include <aquila/source/window/HammingWindow.h>

#include "caffe/common.hpp"
#include "caffe/util/fft.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

struct FastFourierTransformPImpl {
  FastFourierTransformPImpl(int size) : setup(size), window(size) {}
  Aquila::AquilaFft setup;
  Aquila::HammingWindow window;
};

template <>
FastFourierTransform_cpu<double>::FastFourierTransform_cpu(int packetSize, FFTOptions options)
: _log2Size(std::ceil(std::log2(packetSize))),
  _packetSize(static_cast<int>(std::exp2(_log2Size))),
  _pimpl(new FastFourierTransformPImpl(packetSize)),
  _options(options)
{}

template <>
FastFourierTransform_cpu<float>::FastFourierTransform_cpu(int packetSize, FFTOptions options)
: _log2Size(std::ceil(std::log2(packetSize))),
  _packetSize(static_cast<int>(std::exp2(_log2Size))),
  _pimpl(new FastFourierTransformPImpl(packetSize)),
  _options(options)
{}

template <typename Dtype>
FastFourierTransform_cpu<Dtype>::~FastFourierTransform_cpu() {}

template <>
int FastFourierTransform_cpu<double>::process(double* data, int size) {
  CHECK_EQ(size, _packetSize);

  caffe_mul(size, data, _pimpl->window.toArray(), data);

  auto complexData = _pimpl->setup.fft(data);
  for (auto i = 0; i < size / 2; ++i) {
    data[i] = complexData[i].real();
    data[(size / 2) + i] = complexData[i].imag();
  }

  return _packetSize;
}

template <>
int FastFourierTransform_cpu<float>::process(float* data, int size) {
  CHECK_EQ(size, _packetSize);

  auto buffer = vector<double>(size);
  std::copy(data, data + size, std::begin(buffer));

  caffe_mul(size, buffer.data(), _pimpl->window.toArray(), buffer.data());

  auto complexData = _pimpl->setup.fft(buffer.data());
  for (auto i = 0; i < size / 2; ++i) {
    data[i] = complexData[i].real();
    data[(size / 2) + i] = complexData[i].imag();
  }

  return _packetSize;
}

INSTANTIATE_CLASS(FastFourierTransform_cpu);

} // namespace caffe
