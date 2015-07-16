
#include <valarray>
#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/fft.hpp"

namespace caffe {
  
  
FastFourierTransform::FastFourierTransform(int packetSize, FFTOptions options)
: _log2Size(std::ceil(std::log2(packetSize))), _packetSize(static_cast<int>(std::exp2(_log2Size))),
  _window(packetSize),
  _buffer(packetSize),
  _options(options)
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
  
  float* inputBuffer = std::begin(_buffer);
  float* outputBuffer = data;
  
  if (_options.decib()) {
    applyDB(inputBuffer, outputBuffer, _packetSize);
    std::swap(inputBuffer, outputBuffer);
  }

  if (_options.norm()) {
    applyNorm(inputBuffer, outputBuffer, _packetSize);
    std::swap(inputBuffer, outputBuffer);
  }

  if (_options.scale() != 1) {
    applyScale(inputBuffer, outputBuffer, _options.scale(), _packetSize);
    std::swap(inputBuffer, outputBuffer);
  }

  if (_options.polar()) {
    splitData.realp = inputBuffer;
    splitData.imagp = inputBuffer + _packetSize/2;
    applyMagPhase(&splitData, outputBuffer, _packetSize);
    std::swap(inputBuffer, outputBuffer);
  }

  if (inputBuffer != data) {
    std::move(inputBuffer, inputBuffer + _packetSize, data);
  }
  
  return _packetSize;
}

void FastFourierTransform::applyMagPhase(DSPSplitComplex* input, float* output, int size) {
  vDSP_zvmags(input, 1, output, 1, size/2);
  vDSP_zvphas(input, 1, output + size/2, 1, size/2);
}
  
void FastFourierTransform::applyNorm(float* input, float* output, int size) {
  auto mean = 0.0f;
  auto sd = 1.0f;
  vDSP_normalize(input, 1, output, 1, &mean, &sd, size);
}

void FastFourierTransform::applyScale(float* input, float* output, float scale, int size) {
  vDSP_vsmul(input, 1, &scale, output, 1, size);
}
  
void FastFourierTransform::applyDB(float* input, float* output, int size) {
  auto zero = 1.0f;
  vDSP_vdbcon(input, 1, &zero, output, 1, size, 0);
}

} // namespace caffe
