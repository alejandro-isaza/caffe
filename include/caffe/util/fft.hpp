#ifndef CAFFE_UTIL_FFT_HPP
#define CAFFE_UTIL_FFT_HPP

#include <aquila/transform/AquilaFft.h>
#include <aquila/source/window/HammingWindow.h>

#include "caffe/proto/caffe.pb.h"


namespace caffe {

template <typename Dtype>
class FastFourierTransform {
public:
  FastFourierTransform(int packetSize, FFTOptions options);
  ~FastFourierTransform();

  int process(Dtype* data, int size);
  
private:
  const int _log2Size;
  const int _packetSize;
  Aquila::AquilaFft _setup;
  FFTOptions _options;
  Aquila::HammingWindow _window;
};

} // namespace caffe

#endif // CAFFE_UTIL_FFT_HPP
