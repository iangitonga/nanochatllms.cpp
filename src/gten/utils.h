#pragma once

#include <fstream>
#include <chrono>

#include "tensor.h"
#include "gten_types.h"


namespace gten {

const char* dtype_str(Dtype dtype);

void read_into_weight(std::ifstream& fin, gten::Tensor& tensor, ModuleDtype dtype);

void read_layer_header(std::ifstream& fin, bool debug = false);


struct PerformanceMetrics {
    int tokens_generated = 0;
    float throughput_tok_per_sec = 0;
    int inference_total_secs = 0;    
    int sample_time_secs = 0;        
    int load_time_secs = 0;          
    int total_runtime_secs = 0;
    int inference_time_per_tok_ms = 0;
    int linear_time_per_tok_ms = 0;   
    int attn_time_per_tok_ms = 0;     
    int other_time_ms = 0;             
    int mem_usage_total_mb = 0;  
    int mem_usage_weights_mb = 0;
    int mem_usage_acvs_mb = 0;
};

void print_performance_metrics(const PerformanceMetrics& metrics);


class Timer {
public:
    Timer(int* time_tracker);
    ~Timer();
    void stop();
private:
    int* m_time_tracker;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start_time;
    bool m_timer_stopped = false;
};

}
