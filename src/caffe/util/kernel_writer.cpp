//  Copyright (c) 2015 Venture Media. All rights reserved.

#include <fstream>
#include <string>

#include <caffe/caffe.hpp>
#include <caffe/util/kernel_writer.h>

using namespace std;

namespace caffe {
    
int KernelWriter::writeKernelsToTextFile(string modelFilename, string outFilePath) {
    auto net = caffe::NetParameter{};
    caffe::ReadProtoFromBinaryFile(modelFilename, &net);
    
    for (auto i = 0; i < net.layer_size(); ++i) {
        auto layer = net.layer(i);
        if (layer.type() == "Convolution") {
            for (auto j = 0; j < layer.blobs_size(); ++j) {
                auto blob = layer.blobs(j);
                
                auto numKernels = layer.convolution_param().num_output();
                auto kernelWidth = layer.convolution_param().kernel_w();
                
                writeKernelData(outFilePath, blob.data().data(), numKernels, kernelWidth);
            }
        }
    }
    
    return 0;
}

void KernelWriter::writeKernelData(string filePath, const float* data, int numKernels, int kernelWidth) {
    auto fileStreams = openFileStreams(filePath, numKernels);
    
    for (auto k = 0; k < numKernels; ++k) {
        for (auto i = 0; i < kernelWidth; ++i) {
            fileStreams[k] << data[i * k] << endl;
        }
    }
    
    closeFileStreams(fileStreams);
}

vector<fstream> KernelWriter::openFileStreams(string filePath, int num) {
    auto fileStreams = vector<fstream>(num);
    
    for (int i = 0; i < num; ++i) {
        fileStreams[i].open(filePath + to_string(i) + ".txt", fstream::out | fstream::app);
    }
    
    return fileStreams;
}

void KernelWriter::closeFileStreams(vector<fstream>& fileStreams) {
    for (int i = 0; i < fileStreams.size(); ++i) {
        fileStreams[i].close();
    }
}
    
} // namespace caffe
