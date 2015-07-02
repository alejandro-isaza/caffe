
#include <valarray>
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

int FastFourierTransform::process(float* input, int size) {
  CHECK_EQ(size, _packetSize);

  DSPSplitComplex splitData;
  splitData.realp = input;
  splitData.imagp = input + _packetSize/2;

  std::valarray<float> window(size);
  vDSP_hamm_window(std::begin(window), size, 0);

  vDSP_vmul(input, 1, std::begin(window), 1, input, 1, size);
  vDSP_ctoz(reinterpret_cast<const DSPComplex*>(input), 2, &splitData, 1, _packetSize/2);
  vDSP_fft_zrip(_setup, &splitData, 1, _log2Size, kFFTDirection_Forward);

  return _packetSize;
}

} // namespace caffe
