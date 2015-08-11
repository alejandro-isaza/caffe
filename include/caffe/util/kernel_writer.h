#ifndef CAFFE_UTIL_KERNEL_WRITER_HPP
#define CAFFE_UTIL_KERNEL_WRITER_HPP

#include <string>
#include <fstream>
#include <vector>

using namespace std;

namespace caffe {
    
class KernelWriter {
public:
    static int writeKernelsToTextFile(string modelFilename, string outFilePath = "kernel");
    
private:
    static void writeKernelData(string fileName, const float* data, int numKernels, int kernelWidth);
    static vector<fstream> openFileStreams(string filePath, int num);
    static void closeFileStreams(vector<fstream>& fileStreams);
};
    
} // namespace caffe

#endif // CAFFE_UTIL_KERNEL_WRITER_HPP
