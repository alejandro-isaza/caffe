#include "caffe/util/read_audio.hpp"
#include "caffe/common.hpp"
#include <AudioToolbox/AudioToolbox.h>

namespace caffe {

int ReadAudioFile(const std::string& filePath, float* data, int capacity, int offset) {
  CFStringRef sourceFilePath = CFStringCreateWithCStringNoCopy(kCFAllocatorDefault, filePath.c_str(), kCFStringEncodingUTF8, kCFAllocatorDefault);
  CFURLRef sourceURL = CFURLCreateWithFileSystemPath(kCFAllocatorDefault, sourceFilePath, kCFURLPOSIXPathStyle, false);

  ExtAudioFileRef fileRef;
  OSStatus status = ExtAudioFileOpenURL(sourceURL, &fileRef);
  CHECK_EQ(status, noErr) << "Can't open file: " << filePath;

  AudioStreamBasicDescription sourceDescription;
  UInt32 sourceDescriptionSize = sizeof(sourceDescription);
  status = ExtAudioFileGetProperty(fileRef, kExtAudioFileProperty_FileDataFormat, &sourceDescriptionSize, &sourceDescription);
  CHECK_EQ(status, noErr) << "Can't get file description for: " << filePath;

  AudioStreamBasicDescription destinationDescription;
  destinationDescription.mFormatID          = kAudioFormatLinearPCM;
  destinationDescription.mFormatFlags       = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked | kAudioFormatFlagIsNonInterleaved;
  destinationDescription.mChannelsPerFrame  = 1;
  destinationDescription.mBytesPerPacket    = sizeof(Float32);
  destinationDescription.mFramesPerPacket   = 1;
  destinationDescription.mBytesPerFrame     = sizeof(Float32);
  destinationDescription.mBitsPerChannel    = 8 * sizeof(Float32);
  destinationDescription.mSampleRate        = sourceDescription.mSampleRate;
  status = ExtAudioFileSetProperty(fileRef, kExtAudioFileProperty_ClientDataFormat, sizeof(destinationDescription), &destinationDescription);
  CHECK_EQ(status, noErr) << "Can't convert format for: " << filePath;

  status = ExtAudioFileSeek(fileRef, offset);
  CHECK_EQ(status, noErr) << "Can't seek to offset in: " << filePath;

  auto numberOfFrames = static_cast<UInt32>(capacity);

  AudioBufferList bufferList;
  bufferList.mNumberBuffers = 1;
  bufferList.mBuffers[0].mNumberChannels = 1;
  bufferList.mBuffers[0].mDataByteSize = numberOfFrames * sizeof(Float32);
  bufferList.mBuffers[0].mData = data;
  status = ExtAudioFileRead(fileRef, &numberOfFrames, &bufferList);
  CHECK_EQ(status, noErr) << "Can't read file: " << filePath;

  ExtAudioFileDispose(fileRef);
  return numberOfFrames;
}

} // namespace caffe
