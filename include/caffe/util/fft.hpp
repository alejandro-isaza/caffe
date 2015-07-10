#ifndef CAFFE_UTIL_FFT_HPP
#define CAFFE_UTIL_FFT_HPP

#include <valarray>
#include <Accelerate/Accelerate.h>

namespace caffe {

class FastFourierTransform {
public:
  FastFourierTransform(int packetSize);
  ~FastFourierTransform();
  int process(float* data, int size);

private:
  const int _log2Size;
  const int _packetSize;
  FFTSetup _setup;
  std::valarray<float> _window;
  std::valarray<float> _buffer;
};

} // namespace caffe

#endif // CAFFE_UTIL_FFT_HPP
