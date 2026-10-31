#include "demucs.hpp"
#include "dsp.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <unsupported/Eigen/FFT>
#include <unsupported/Eigen/MatrixFunctions>
#include <vector>
#include <Eigen/Dense>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include "demucs.hpp"

namespace demucsonnx {
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::RunOptions run_options;
}

// Core function to create and load model from in-memory data (byte array)
bool demucsonnx::load_model(
    const char* model_data,
    int n_bytes,
    struct demucsonnx::demucs_model &model,
    Ort::SessionOptions &session_options)
{
    const uint8_t* final_data = reinterpret_cast<const uint8_t*>(model_data);
    size_t final_size = n_bytes;

    model.sess = std::make_unique<Ort::Session>(model.env, final_data, final_size, session_options);

    std::vector<Ort::AllocatedStringPtr> input_name_allocs;
    input_name_allocs.push_back(model.sess->GetInputNameAllocated(0, demucsonnx::allocator));

    model.input_names.push_back(input_name_allocs[0].get());  // Store as std::string

    std::vector<Ort::AllocatedStringPtr> output_name_allocs;
    output_name_allocs.push_back(model.sess->GetOutputNameAllocated(0, demucsonnx::allocator));

    model.output_names.push_back(output_name_allocs[0].get());

    for (const auto& name : model.input_names) {
        model.input_names_ptrs.push_back(name.c_str());
    }
    for (const auto& name : model.output_names) {
        model.output_names_ptrs.push_back(name.c_str());
    }

    auto output0_shape = model.sess->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

    model.nb_sources = output0_shape[1];
    return true;
}

// Overload for std::vector<char>
bool demucsonnx::load_model(
    const std::vector<char> &model_data,
    struct demucsonnx::demucs_model &model,
    Ort::SessionOptions &session_options)
{
    return load_model(
        model_data.data(), model_data.size(), model, session_options);
}

void RunONNXInference(
    struct demucsonnx::demucs_model &model,
    struct demucsonnx::demucs_segment_buffers &buffers
) {
    // Run the model
    model.sess->Run(
        demucsonnx::run_options,
        model.input_names_ptrs.data(),
        buffers.input_tensors.data(),
        buffers.input_tensors.size(),
        model.output_names_ptrs.data(),
        buffers.output_tensors.data(),
        model.output_names_ptrs.size()
    );
}

// run core demucs inference using onnx
void demucsonnx::model_inference(
    struct demucsonnx::demucs_model &model,
    struct demucsonnx::demucs_segment_buffers &buffers)
    // struct demucsonnx::stft_buffers &stft_buf)
{

    // prepare time branch input by copying buffers.mix into  input_tensors[0]
    float *xt_onnx_data = buffers.input_tensors[0].GetTensorMutableData<float>();

    for (int i = 0; i < buffers.padded_mix.rows(); ++i)
    {
        for (int j = 0; j < buffers.segment_samples; ++j)
        {
            // calculate destination index, simple row major calculation
            // given the onnx shape of (1, 2, segment_samples)
            int dest_index = i * buffers.segment_samples + j;
            xt_onnx_data[dest_index] = buffers.padded_mix(i, j + buffers.pad);
        }
    }
    
    // now we apply the core demucs inference
    RunONNXInference(model, buffers);

    std::cout << "ONNX inference completed." << std::endl;

    int nb_out_sources = model.nb_sources;

    // nb_sources sources, 2 channels, N samples
    std::vector<Eigen::MatrixXf> xt_3d(
        nb_out_sources,
        Eigen::MatrixXf(2, buffers.segment_samples)
    );

    // Map output onnx tensors
    float* xt_out_data = buffers.output_tensors[0].GetTensorMutableData<float>();

    for (int s = 0; s < nb_out_sources; ++s)
    { // loop over 4 sources
        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < buffers.segment_samples; ++j)
            {
                int index = s * 2 * buffers.segment_samples + i * buffers.segment_samples + j;
                buffers.targets_out(s, i, j) = xt_out_data[index];
            }
        }
    }
}
