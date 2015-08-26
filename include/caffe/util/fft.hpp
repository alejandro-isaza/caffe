#ifndef CAFFE_UTIL_FFT_HPP
#define CAFFE_UTIL_FFT_HPP

#include "caffe/proto/caffe.pb.h"


namespace caffe {

struct FastFourierTransformPImpl;

template <typename Dtype>
class FastFourierTransform_cpu {
public:
  FastFourierTransform_cpu(int packetSize, FFTOptions options);
  ~FastFourierTransform_cpu();

  int process(Dtype* data, int size);

private:
  const int _log2Size;
  const int _packetSize;

  std::unique_ptr<FastFourierTransformPImpl> _pimpl;

  FFTOptions _options;
};

template <typename Dtype>
class FastFourierTransform_gpu {
public:
  FastFourierTransform_gpu(int packetSize, FFTOptions options);
  ~FastFourierTransform_gpu();

  int process(Dtype* data, int size);

private:
  const int _log2Size;
  const int _packetSize;

  std::unique_ptr<FastFourierTransformPImpl> _pimpl;

  FFTOptions _options;
};

} // namespace caffe

#endif // CAFFE_UTIL_FFT_HPP
