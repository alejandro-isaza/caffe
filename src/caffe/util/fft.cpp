
#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/fft.hpp"

namespace caffe {

FastFourierTransform::FastFourierTransform(int packetSize)
    : _log2Size(std::ceil(std::log2(packetSize))), _packetSize(static_cast<int>(std::exp2(_log2Size))) {
  _setup = vDSP_create_fftsetup(_log2Size, kFFTRadix2);
}

FastFourierTransform::~FastFourierTransform() {
  vDSP_destroy_fftsetup(_setup);
}

int FastFourierTransform::process(const float* input, int size, float* output, int capacity) {
  CHECK_EQ(size, _packetSize);

  DSPSplitComplex splitData;
  splitData.realp = output;
  splitData.imagp = output + _packetSize/2;

  vDSP_ctoz(reinterpret_cast<const DSPComplex*>(input), 2, &splitData, 1, _packetSize/2);
  vDSP_fft_zrip(_setup, &splitData, 1, _log2Size, kFFTDirection_Forward);

  // Scale so that the values are between -1 and 1
  float scaleFactor = 1.0f / _packetSize;
  vDSP_vsmul(output, 1, &scaleFactor, output, 1, _packetSize);

  return _packetSize;
}

} // namespace caffe
