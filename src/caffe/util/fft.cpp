
#include <valarray>
#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/fft.hpp"

namespace caffe {

FastFourierTransform::FastFourierTransform(int packetSize)
: _log2Size(std::ceil(std::log2(packetSize))), _packetSize(static_cast<int>(std::exp2(_log2Size))),
  _window(packetSize),
  _buffer(packetSize)
{
  _setup = vDSP_create_fftsetup(_log2Size, kFFTRadix2);
  vDSP_hamm_window(std::begin(_window), _packetSize, 0);
}

FastFourierTransform::~FastFourierTransform() {
  vDSP_destroy_fftsetup(_setup);
}

int FastFourierTransform::process(float* data, int size) {
  CHECK_EQ(size, _packetSize);

  vDSP_vmul(data, 1, std::begin(_window), 1, data, 1, size);

  DSPSplitComplex splitData;
  splitData.realp = std::begin(_buffer);
  splitData.imagp = std::begin(_buffer) + _packetSize/2;
  vDSP_ctoz(reinterpret_cast<const DSPComplex*>(data), 2, &splitData, 1, _packetSize/2);

  vDSP_fft_zrip(_setup, &splitData, 1, _log2Size, kFFTDirection_Forward);

  vDSP_zvmags(&splitData, 1, data, 1, _packetSize/2);
  vDSP_zvphas(&splitData, 1, data + _packetSize/2, 1, _packetSize/2);

  return _packetSize;
}

} // namespace caffe
