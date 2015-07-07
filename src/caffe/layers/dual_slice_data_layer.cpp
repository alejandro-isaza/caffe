#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
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
            lines_.push_back(std::make_tuple<std::string, std::string, int, int, int>(element["file1"], element["file2"], element["label"], element["offset1"], element["offset2"]));
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
        datum.set_height(1);
        datum.set_width(this->layer_param_.dual_slice_data_param().width());

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
        const auto width = this->layer_param_.dual_slice_data_param().width();
        const auto singleFileWidth = width / 2;


        Datum datum;
        datum.set_channels(1);
        datum.set_height(1);
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

            const auto label = std::get<2>(lines_[lines_id_]);
            const auto fileNames = std::vector<std::string>{std::get<0>(lines_[lines_id_]), std::get<1>(lines_[lines_id_])};
            const auto offsets = std::vector<int>{std::get<3>(lines_[lines_id_]), std::get<4>(lines_[lines_id_])};

            Datum datum;
            datum.set_channels(1);
            datum.set_height(1);
            datum.set_width(width);
            datum.set_label(label);

            std::vector<float> tempData(singleFileWidth);

            for (int i = 0; i < 2; ++i) {
                ReadAudioFile(root_folder + fileNames[i], &tempData.front(), tempData.size(), offsets[i]);
                if (this->layer_param_.dual_slice_data_param().fft()) {
                    auto fft = FastFourierTransform(singleFileWidth);
                    fft.process(&tempData.front(), tempData.size());
                }
                for (int i = 0; i < singleFileWidth; i++) {
                    datum.add_float_data(tempData[i]);
                }
            }



            read_time += timer.MicroSeconds();
            timer.Start();
            // Apply transformations to the audio
            int offset = this->prefetch_data_.offset(item_id);
            this->transformed_data_.set_cpu_data(prefetch_data + offset);
            this->data_transformer_->Transform(datum, &(this->transformed_data_));
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
    
    INSTANTIATE_CLASS(DualSliceDataLayer);
    REGISTER_LAYER_CLASS(DualSliceData);
    
}  // namespace caffe
