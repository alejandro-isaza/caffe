#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/dsp.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/read_audio.hpp"
#include "caffe/util/fft.hpp"
#include "json/json.hpp"

namespace caffe {

    template <typename Dtype>
    DualSliceDataLayer<Dtype>::~DualSliceDataLayer<Dtype>() {
        this->JoinPrefetchThread();
    }

    template <typename Dtype>
    void DualSliceDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top) {
        string root_folder = this->layer_param_.dual_slice_data_param().root_folder();

        // Read the file with filenames and labels
        const string& source = this->layer_param_.dual_slice_data_param().source();
        LOG(INFO) << "Opening file " << source;

        std::ifstream ifs(source.c_str());
        nlohmann::json json;
        json << ifs;

        for (auto element : json) {
            auto data = AudioData{};
            data.file1 = element["file1"].get<std::string>();
            data.file2 = element["file2"].get<std::string>();
            data.label = element["label"].get<int>();
            data.offset1 = element["offset1"].get<int>();
            data.offset2 = element["offset2"].get<int>();
            lines_.push_back(std::move(data));
        }

        if (this->layer_param_.dual_slice_data_param().shuffle()) {
            // randomly shuffle data
            LOG(INFO) << "Shuffling data";
            const unsigned int prefetch_rng_seed = caffe_rng_rand();
            prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
            ShuffleFiles();
        }
        LOG(INFO) << "A total of " << lines_.size() << " files.";

        lines_id_ = 0;

        Datum datum;
        datum.set_channels(1);
        datum.set_height(2);
        datum.set_width(this->layer_param_.dual_slice_data_param().sample_count());

        // Use data_transformer to infer the expected blob shape from a datum.
        vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
        this->transformed_data_.Reshape(top_shape);
        // Reshape prefetch_data and top[0] according to the batch_size.
        const int batch_size = this->layer_param_.dual_slice_data_param().batch_size();
        top_shape[0] = batch_size;
        this->prefetch_data_.Reshape(top_shape);
        top[0]->ReshapeLike(this->prefetch_data_);

        LOG(INFO) << "output data size: " << top[0]->num() << ","
        << top[0]->channels() << "," << top[0]->height() << ","
        << top[0]->width();
        // label
        vector<int> label_shape(1, batch_size);
        top[1]->Reshape(label_shape);
        this->prefetch_label_.Reshape(label_shape);
    }

    template <typename Dtype>
    void DualSliceDataLayer<Dtype>::ShuffleFiles() {
        caffe::rng_t* prefetch_rng =
        static_cast<caffe::rng_t*>(prefetch_rng_->generator());
        shuffle(lines_.begin(), lines_.end(), prefetch_rng);
    }

    // This function is used to create a thread that prefetches the data.
    template <typename Dtype>
    void DualSliceDataLayer<Dtype>::InternalThreadEntry() {
        CPUTimer batch_timer;
        batch_timer.Start();
        double read_time = 0;
        double trans_time = 0;
        CPUTimer timer;
        CHECK(this->prefetch_data_.count());
        CHECK(this->transformed_data_.count());
        DualSliceDataParameter dual_slice_data_param = this->layer_param_.dual_slice_data_param();
        const int batch_size = dual_slice_data_param.batch_size();
        string root_folder = dual_slice_data_param.root_folder();
        const auto sample_count = this->layer_param_.dual_slice_data_param().sample_count();
        const auto width = static_cast<int>(sample_count);
        const auto height = 2;

        Datum datum;
        datum.set_channels(1);
        datum.set_height(height);
        datum.set_width(width);

        // Use data_transformer to infer the expected blob shape from a datum.
        vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
        this->transformed_data_.Reshape(top_shape);
        // Reshape prefetch_data according to the batch_size.
        top_shape[0] = batch_size;
        this->prefetch_data_.Reshape(top_shape);

        Dtype* prefetch_data = this->prefetch_data_.mutable_cpu_data();
        Dtype* prefetch_label = this->prefetch_label_.mutable_cpu_data();

        // datum scales
        const int lines_size = lines_.size();
        for (int item_id = 0; item_id < batch_size; ++item_id) {
            // get a blob
            timer.Start();

            const auto label = lines_[lines_id_].label;
            const auto fileNames = std::vector<std::string>{lines_[lines_id_].file1, lines_[lines_id_].file2};
            const auto offsets = std::vector<int>{lines_[lines_id_].offset1 + shiftDistribution(prng), lines_[lines_id_].offset2 + shiftDistribution(prng)};
            const auto gain = std::exp(gainDistribution(prng));
            
            Blob<Dtype> blob({1, 1, height, width});
            auto data = blob.mutable_cpu_data();

            fetchFFTransformedData(root_folder + fileNames[0], data, offsets[0], gain, sample_count);
            fetchFFTransformedData(root_folder + fileNames[1], data + sample_count, offsets[1], gain, sample_count);

            read_time += timer.MicroSeconds();
            timer.Start();
            // Apply transformations to the audio
            int offset = this->prefetch_data_.offset(item_id);
            this->transformed_data_.set_cpu_data(prefetch_data + offset);
            this->data_transformer_->Transform(&blob, &(this->transformed_data_));
            trans_time += timer.MicroSeconds();
            
            prefetch_label[item_id] = label;
            // go to the next iter
            lines_id_++;
            if (lines_id_ >= lines_size) {
                // We have reached the end. Restart from the first.
                DLOG(INFO) << "Restarting data prefetching from start.";
                lines_id_ = 0;
                if (this->layer_param_.dual_slice_data_param().shuffle()) {
                    ShuffleFiles();
                }
            }
        }
        batch_timer.Stop();
        DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
        DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
        DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
    }

    template <typename Dtype>
    void DualSliceDataLayer<Dtype>::fetchFFTransformedData(const std::string& filename, Dtype* data, int offset, Dtype gain, int size) {
        ReadAudioFile(filename, data, size, offset);
        dsp::vsmul(data, 1, &gain, data, 1, size);

        if (this->layer_param_.dual_slice_data_param().fft()) {
            auto fft = FastFourierTransform<Dtype>(size, this->layer_param_.dual_slice_data_param().fft_options());
            fft.process(data, size);
        }
    }

    
    INSTANTIATE_CLASS(DualSliceDataLayer);
    REGISTER_LAYER_CLASS(DualSliceData);
    
}  // namespace caffe
