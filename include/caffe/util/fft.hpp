#ifndef CAFFE_UTIL_FFT_HPP
#define CAFFE_UTIL_FFT_HPP

#include <valarray>

#include "caffe/util/dsp.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

template <typename Dtype>
class FastFourierTransform {
public:
  FastFourierTransform(dsp::Length packetSize, FFTOptions options);
  ~FastFourierTransform();

  int process(Dtype* data, dsp::Length size);

private:
  void applyMagPhase(dsp::SplitComplex<Dtype>* input, Dtype* output, dsp::Length size);
  void applyNorm(Dtype* input, Dtype* output, dsp::Length size);
  void applyScale(Dtype* input, Dtype* output, Dtype scale, dsp::Length size);
  void applyDB(Dtype* input, Dtype* output, dsp::Length size);
  
private:
  const dsp::Length _log2Size;
  const dsp::Length _packetSize;

  dsp::FFTSetup<Dtype> _setup;
  FFTOptions _options;
  std::valarray<Dtype> _window;
  std::valarray<Dtype> _buffer;
};

} // namespace caffe

#endif // CAFFE_UTIL_FFT_HPP
