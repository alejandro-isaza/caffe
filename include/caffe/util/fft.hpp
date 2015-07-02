#ifndef CAFFE_UTIL_FFT_HPP
#define CAFFE_UTIL_FFT_HPP

#include <Accelerate/Accelerate.h>

namespace caffe {

class FastFourierTransform {
public:
  FastFourierTransform(int packetSize);
  ~FastFourierTransform();
  int process(float* input, int size);

private:
  const int _log2Size;
  const int _packetSize;
  FFTSetup _setup;
};

} // namespace caffe

#endif // CAFFE_UTIL_FFT_HPP
