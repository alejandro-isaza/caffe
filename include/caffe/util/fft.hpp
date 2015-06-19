#include "caffe.hpp"
#include <Accelerate/Accelerate.h>

namespace caffe {
class FastFourierTransform {
public:
  FastFourierTransform(int packetSize);
  ~FastFourierTransform();
  int process(const float* input, int size, float* output, int capacity);

private:
  const int _log2Size;
  const int _packetSize;
  FFTSetup _setup;
};
}