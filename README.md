 # nanochatllms.cpp

**nanochatllms.cpp** is a repository containing pure C++ implementations of Chat-LLMs
with less than 3 billion parameters. The goal is to provide implementation of quantised
small Chat-LLMs that can run efficiently on lower-end devices. The models are
implemented in fp16, 8-bit and 4-bit formats. This project was inspired by
[llama.cpp](https://github.com/ggerganov/llama.cpp) and [llama.c](https://github.com/karpathy/llama2.c)

## Implemented models
1. MiniCPM-2B DPO [source](https://huggingface.co/openbmp/MiniCPM-2B-dpo-fp16) [License](https://github.com/OpenBMB/General-Model-License/blob/main/%E9%80%9A%E7%94%A8%E6%A8%A1%E5%9E%8B%E8%AE%B8%E5%8F%AF%E5%8D%8F%E8%AE%AE-%E6%9D%A5%E6%BA%90%E8%AF%B4%E6%98%8E-%E5%AE%A3%E4%BC%A0%E9%99%90%E5%88%B6-%E5%95%86%E4%B8%9A%E6%8E%88%E6%9D%83.md)
1. StableLM-2-Zephyr-1.6B [source](https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b)
2. TinyLlama-1.1B-Chat-v0.4 [source](https://github.com/jzhang38/TinyLlama)


## Build and run
```
git clone https://github.com/iangitonga/nanochatllms.cpp
cmake -S . -B build/
cmake --build build/ --config Release
build/bin/nanochat -m tinyllama -p "Give three tips on staying healthy."
```

To see all the available options, run
```
build/bin/nanochat --help
```

## Sample Metrics

**Note:** Performance was recorded on a Intel-Xeon CPU @ 2.20GHz with two cores with AVX enabled.

| Model          | Format | Model size (GB) | Performance (tokens/sec) |
| -------------- | ------ | --------------- | ------------------------ |
| MiniCPM-2B     | FP16   | 5.5             |  2.4                     |
|                | Q8     | 2.9             |  3.2                     |
|                | Q4     | 1.5             |  3.4                     |
| Zephyr-1.6B    | FP16   | 3.3             |  4.2                     |
|                | Q8     | 1.8             |  5.2                     |
|                | Q4     | 0.9             |  5.6                     |
| TinyLlama-1.1B | FP16   | 2.2             |  6.3                     |
|                | Q8     | 1.2             |  8.7                     |
|                | Q4     | 0.6             |  9.1                     |
