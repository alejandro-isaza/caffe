
#include <valarray>
#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/fft.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <>
FastFourierTransform<double>::FastFourierTransform(int packetSize, FFTOptions options)
: _log2Size(std::ceil(std::log2(packetSize))),
  _packetSize(static_cast<int>(std::exp2(_log2Size))),
  _window(packetSize),
  _options(options),
  _setup(packetSize)
{
  _window = Aquila::HammingWindow(_packetSize);
}
  
template <>
FastFourierTransform<float>::FastFourierTransform(int packetSize, FFTOptions options)
: _log2Size(std::ceil(std::log2(packetSize))),
  _packetSize(static_cast<int>(std::exp2(_log2Size))),
  _window(packetSize),
  _options(options),
  _setup(packetSize)
{
  _window = Aquila::HammingWindow(_packetSize);
}

template <typename Dtype>
FastFourierTransform<Dtype>::~FastFourierTransform() {}

template <>
int FastFourierTransform<double>::process(double* data, int size) {
  CHECK_EQ(size, _packetSize);

  caffe_mul(size, data, _window.toArray(), data);

  auto complexData = _setup.fft(data);
  for (auto i = 0; i < size / 2; ++i) {
    data[i] = complexData[i].real();
    data[(size / 2) + i] = complexData[i].imag();
  }

  return _packetSize;
}

template <>
int FastFourierTransform<float>::process(float* data, int size) {
  CHECK_EQ(size, _packetSize);
  
  auto buffer = vector<double>(size);
  std::copy(data, data + size, std::begin(buffer));
  
  caffe_mul(size, buffer.data(), _window.toArray(), buffer.data());
  
  auto complexData = _setup.fft(buffer.data());
  for (auto i = 0; i < size / 2; ++i) {
    data[i] = complexData[i].real();
    data[(size / 2) + i] = complexData[i].imag();
  }
  
  return _packetSize;
}

INSTANTIATE_CLASS(FastFourierTransform);

} // namespace caffe
