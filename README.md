 # nanochatllms.cpp

**nanochatllms.cpp** is a repository containing pure C++ implementations of Chat-LLMs
with less than 3 billion parameters. The goal is to provide implementation of quantised
small Chat-LLMs that can run efficiently on lower-end devices offline. The models are
implemented in fp16, 8-bit and 4-bit formats. This project was inspired by
[llama.cpp](https://github.com/ggerganov/llama.cpp) and [llama.c](https://github.com/karpathy/llama2.c)

## Implemented models
1. TinyLlama-1.1B-Chat-v0.4
2. Zephyr1.6B


## Metrics

| Model | Format | Model size | Performance (tokens/sec) |
| ----- | ------ | ---------- | ------------------------ |
| Zephyr1.6b | FP16 | 3.29 GB |  10                      |
|            | Q8   | 1.75 GB |  10                      |
|            | Q4   | 0.92 GB |  10                      |
| TinyLlama1.1B | FP16 | 2.2 GB  |  10                   |
|               | Q8   | 1.17 GB |  10                   |
|               | Q4   | 0.6 GB  |  10                   |
