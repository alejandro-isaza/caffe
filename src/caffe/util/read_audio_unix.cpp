//  Created by Aidan Gomez on 2015-08-19.
//  Copyright (c) 2015 Venture Media. All rights reserved.

#include "caffe/util/read_audio.hpp"
#include "caffe/common.hpp"

#include <sndfile.h>

namespace caffe {
    
    int ReadAudioFile(const std::string& filePath, float* data, int capacity, int offset) {
        auto info = SF_INFO{};
        
        auto file = sf_open(filePath.c_str(), SFM_READ, &info);
        CHECK_EQ(sf_error(file), SF_ERR_NO_ERROR) << "Can't open file '" << filePath << "': " << sf_strerror(file);

        auto status = sf_seek(file, offset, SEEK_SET);
        CHECK_NE(status, -1) << "Can't seek to offset in: '" << filePath << "': " << sf_strerror(file);
        
        auto numberOfFrames = sf_read_float(file, data, capacity);
        CHECK_EQ(numberOfFrames, capacity) << "File could not fill provided array";
        
        return numberOfFrames;
    }
    
    int ReadAudioFile(const std::string& filePath, double* data, int capacity, int offset) {
        auto info = SF_INFO{};
        
        auto file = sf_open(filePath.c_str(), SFM_READ, &info);
        CHECK_EQ(sf_error(file), SF_ERR_NO_ERROR) << "Can't open file '" << filePath << "': " << sf_strerror(file);
        
        auto status = sf_seek(file, offset, SEEK_SET);
        CHECK_NE(status, -1) << "Can't seek to offset in: '" << filePath << "': " << sf_strerror(file);
        
        auto numberOfFrames = sf_read_double(file, data, capacity);
        CHECK_EQ(numberOfFrames, capacity) << "File could not fill provided array";
        
        return numberOfFrames;
    }
    
} // namespace caffe
