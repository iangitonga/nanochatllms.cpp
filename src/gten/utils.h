#pragma once

#include <fstream>
#include <chrono>

#include "tensor.h"
#include "gten_types.h"


namespace gten {

const char* dtype_str(Dtype dtype);

void read_into_weight(std::ifstream& fin, gten::Tensor& tensor, ModuleDtype dtype);

void read_layer_header(std::ifstream& fin, bool debug = false);


class Timer {
public:
    Timer(int64_t* time_tracker);
    ~Timer();
    void stop();
private:
    int64_t* time_tracker_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
    bool stopped_ = false;
};

}
