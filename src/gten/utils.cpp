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

Timer::Timer(int64_t* time_tracker)
    : time_tracker_{time_tracker}, start_time_{std::chrono::high_resolution_clock::now()}
{ 
}

Timer::~Timer() { stop(); }

void Timer::stop() {
    if (stopped_)
        return;
    auto end_time = std::chrono::high_resolution_clock::now();
    int64_t start = std::chrono::time_point_cast<std::chrono::milliseconds>(start_time_).time_since_epoch().count();
    int64_t end = std::chrono::time_point_cast<std::chrono::milliseconds>(end_time).time_since_epoch().count();
    int64_t duration = end - start;
    *time_tracker_ += duration;
    stopped_ = true;
}

} // namespace gten
