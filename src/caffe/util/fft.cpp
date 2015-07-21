
#include <valarray>
#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/fft.hpp"

namespace caffe {

template <typename Dtype>
FastFourierTransform<Dtype>::FastFourierTransform(dsp::Length packetSize, FFTOptions options)
: _log2Size(std::ceil(std::log2(packetSize))), _packetSize(static_cast<int>(std::exp2(_log2Size))),
  _window(packetSize),
  _buffer(packetSize),
  _options(options)
{
  _setup = dsp::create_fftsetup<Dtype>(_log2Size, dsp::FFTRadix::Radix2);
  dsp::hamm_window(std::begin(_window), _packetSize, 0);
}

template <typename Dtype>
FastFourierTransform<Dtype>::~FastFourierTransform() {
  dsp::destroy_fftsetup(_setup);
}

template <typename Dtype>
int FastFourierTransform<Dtype>::process(Dtype* data, dsp::Length size) {
  CHECK_EQ(size, _packetSize);

  dsp::vmul(data, 1, std::begin(_window), 1, data, 1, size);

  dsp::SplitComplex<Dtype> splitData;
  splitData.realp = std::begin(_buffer);
  splitData.imagp = std::begin(_buffer) + _packetSize/2;
  dsp::ctoz(reinterpret_cast<const dsp::Complex<Dtype>*>(data), 2, &splitData, 1, _packetSize/2);

  dsp::fft_zrip(_setup, &splitData, 1, _log2Size, dsp::FFTDirection::Forward);
  
  Dtype* inputBuffer = std::begin(_buffer);
  Dtype* outputBuffer = data;

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

    if (_options.decib()) {
      applyDB(inputBuffer, outputBuffer, _packetSize/2);

      // Mmove the phases without chaning them
      std::move(inputBuffer + _packetSize/2, inputBuffer + _packetSize, outputBuffer + _packetSize/2);

      std::swap(inputBuffer, outputBuffer);
    }
  }

  if (inputBuffer != data) {
    std::move(inputBuffer, inputBuffer + _packetSize, data);
  }
  
  return _packetSize;
}

template <typename Dtype>
void FastFourierTransform<Dtype>::applyMagPhase(dsp::SplitComplex<Dtype>* input, Dtype* output, dsp::Length size) {
  dsp::zvmags(input, 1, output, 1, size/2);
  dsp::zvphas(input, 1, output + size/2, 1, size/2);
}

template <typename Dtype>
void FastFourierTransform<Dtype>::applyNorm(Dtype* input, Dtype* output, dsp::Length size) {
  auto mean = Dtype(0);
  auto sd = Dtype(1);
  dsp::normalize(input, 1, output, 1, &mean, &sd, size);
}

template <typename Dtype>
void FastFourierTransform<Dtype>::applyScale(Dtype* input, Dtype* output, Dtype scale, dsp::Length size) {
  dsp::vsmul(input, 1, &scale, output, 1, size);
}

template <typename Dtype>
void FastFourierTransform<Dtype>::applyDB(Dtype* input, Dtype* output, dsp::Length size) {
  auto zeroRef = Dtype(0);
  dsp::vdbcon(input, 1, &zeroRef, output, 1, size, 1);
}

INSTANTIATE_CLASS(FastFourierTransform);

} // namespace caffe
