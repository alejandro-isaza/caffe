#ifndef CAFFE_UTIL_FFT_HPP
#define CAFFE_UTIL_FFT_HPP

#include <valarray>
#include <Accelerate/Accelerate.h>

#include "caffe/proto/caffe.pb.h"

namespace caffe {

  
class FastFourierTransform {
public:
  FastFourierTransform(int packetSize, FFTOptions options);
  ~FastFourierTransform();
  int process(float* data, int size);
private:
  void applyMagPhase(DSPSplitComplex* input, float* output, int size);
  void applyNorm(float* input, float* output, int size);
  void applyScale(float* input, float* output, float scale, int size);
  void applyDB(float* input, float* output, int size);
  
private:
  const int _log2Size;
  const int _packetSize;
  FFTSetup _setup;
  FFTOptions _options;
  std::valarray<float> _window;
  std::valarray<float> _buffer;
};

} // namespace caffe

#endif // CAFFE_UTIL_FFT_HPP
