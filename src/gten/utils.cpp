#include <iomanip>

#include "gten_types.h"
#include "utils.h"


namespace gten {


const char* dtype_str(Dtype dtype) {
    switch (dtype) {
        case kInt32: 
            return "Int32";
        case kQint8:
            return "Qint8";
        case kQint4:
            return "Qint4";
        case kFloat16:
            return "Float16";
        case kFloat32:
            return "Float32";
        default: {
            GTEN_ASSERT(false);
            return "";
        }
    }
}

void read_into_weight(std::ifstream& fin, gten::Tensor& tensor, ModuleDtype dtype)
{
    std::string weight_name;
    int32_t weight_name_size;
    fin.read(reinterpret_cast<char*>(&weight_name_size), sizeof(weight_name_size));
    weight_name.resize(weight_name_size);
    fin.read(reinterpret_cast<char*>(weight_name.data()), weight_name_size);

    int32_t weight_payload_size;
    fin.read(reinterpret_cast<char*>(&weight_payload_size), sizeof(weight_payload_size));

    // if (debug)
        // std::cout << weight_name << " (" << weight_payload_size << ")\n";

    GTEN_ASSERTM(
        static_cast<size_t>(weight_payload_size) == tensor.nbytes(),
        "Weight `%s` data size: %d does not match the expected size: %ld.",
        weight_name.c_str(), weight_payload_size, tensor.nbytes());
    fin.read(tensor.data_ptr<char>(), weight_payload_size);
}


void read_layer_header(std::ifstream& fin, bool debug) {
    std::string layer_name;
    int32_t layer_name_size;
    fin.read(reinterpret_cast<char*>(&layer_name_size), sizeof(layer_name_size));
    layer_name.resize(layer_name_size);
    fin.read(reinterpret_cast<char*>(layer_name.data()), layer_name_size);

    if (debug) {
        std::cout << "Layer: " << layer_name << "\n";
    }
}

Timer::Timer(int* time_tracker)
    : m_time_tracker{time_tracker}, m_start_time{std::chrono::high_resolution_clock::now()}
{ 
}

Timer::~Timer() { stop(); }

void Timer::stop() {
    if (m_timer_stopped)
        return;
    auto end_time = std::chrono::high_resolution_clock::now();
    int64_t start = std::chrono::time_point_cast<std::chrono::milliseconds>(m_start_time).time_since_epoch().count();
    int64_t end = std::chrono::time_point_cast<std::chrono::milliseconds>(end_time).time_since_epoch().count();
    int64_t duration = end - start;
    *m_time_tracker += static_cast<int>(duration);
    m_timer_stopped = true;
}


void print_performance_metrics(const PerformanceMetrics& metrics)
{
    std::cout << "\n---------------------------------------\n";
    std::cout << " " << "PERFORMANCE\n";
    std::cout << "---------------------------------------\n";
    std::cout << " " << "Tokens generated         : " << std::setw(4) << metrics.tokens_generated        << "\n";
    std::cout << " " << "Throughput (toks/sec)    : " << std::fixed << std::setprecision(1) << std::setw(4) << metrics.throughput_tok_per_sec  << "\n";
    std::cout << " " << "Sample time              : " << std::setw(4) << metrics.sample_time_secs        << "s\n";
    std::cout << " " << "Load time                : " << std::setw(4) << metrics.load_time_secs          << "s\n";
    std::cout << " " << "Inference time           : " << std::setw(4) << metrics.inference_total_secs    << "s\n";
    std::cout << " " << "Total runtime            : " << std::setw(4) << metrics.total_runtime_secs      << "s\n";
    std::cout << "---------------------------------------\n";
    std::cout << " " << "Mem usage (total)        : " << std::setw(4) << metrics.mem_usage_total_mb   << "MB\n";
    std::cout << " " << "Mem usage (weights)      : " << std::setw(4) << metrics.mem_usage_weights_mb << "MB\n";
    std::cout << " " << "Mem usage (actvs)        : " << std::setw(4) << metrics.mem_usage_acvs_mb    << "MB\n";
    std::cout << "---------------------------------------\n";
    std::cout << " " << "Inference time (per tok) : " << std::setw(4) << metrics.inference_time_per_tok_ms << "ms\n";
    std::cout << " " << "Lin time       (per tok) : " << std::setw(4) << metrics.linear_time_per_tok_ms    << "ms\n";
    std::cout << " " << "Attn time      (per tok) : " << std::setw(4) << metrics.attn_time_per_tok_ms      << "ms\n";
    std::cout << " " << "Other          (per tok) : " << std::setw(4) << metrics.other_time_ms             << "ms\n";
    std::cout << "---------------------------------------\n\n";
}

} // namespace gten
