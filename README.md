 # nanochatllms.cpp

**nanochatllms.cpp** is a repository containing pure C++ implementations of Chat-LLMs
with less than 3 billion parameters. The goal is to provide implementation of quantised
small Chat-LLMs that can run efficiently on lower-end devices offline. The models are
implemented in fp16, 8-bit and 4-bit formats. This project was inspired by
[llama.cpp](https://github.com/ggerganov/llama.cpp) and [llama.c](https://github.com/karpathy/llama2.c)

## Implemented models
1. Zephyr1.6B
2. TinyLlama-1.1B-Chat-v0.4


## Metrics

**Note:** Performance was recorded on a Intel(R) Xeon(R) CPU @ 2.20GHz with
two cores.

| Model | Format | Model size | Performance (tokens/sec) |
| ----- | ------ | ---------- | ------------------------ |
|            | Q8   | 1.75 GB |  5.2                     |
|            | Q4   | 0.92 GB |  5.58                    |
| TinyLlama1.1B | FP16 | 2.2 GB  |  6.3                  |
|               | Q8   | 1.17 GB |  8.7                  |
|               | Q4   | 0.6 GB  |  9.1                  |
