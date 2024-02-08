# MiniCPM

**MiniCPM** is an implementation of [MiniCPM-2B DPO](https://huggingface.co/openbmp/MiniCPM-2B-dpo-fp16).
Supported formats are FP16 (5.5GB), 8-bit (2.9GB) and 4-bit (1.5GB).
Model weights for MiniCPM are licensed under [General-Model-License](https://github.com/OpenBMB/General-Model-License/blob/main/%E9%80%9A%E7%94%A8%E6%A8%A1%E5%9E%8B%E8%AE%B8%E5%8F%AF%E5%8D%8F%E8%AE%AE-%E6%9D%A5%E6%BA%90%E8%AF%B4%E6%98%8E-%E5%AE%A3%E4%BC%A0%E9%99%90%E5%88%B6-%E5%95%86%E4%B8%9A%E6%8E%88%E6%9D%83.md)


## Install and Chat with MiniCPM.
```
git clone https://github.com/iangitonga/nanochatllms.cpp
cd nanochatllms.cpp/miniCPM/
g++ -std=c++17 -I../ -O3 -fopenmp -ffast-math minicpm.cpp -o minicpm
./minicpm
```

If you have an Intel CPU that supports AVX and f16c compile with the following
 command to achieve higher performance:

```
g++ -std=c++17 -I../ -O3 -fopenmp -ffast-math -mavx -mf16c minicpm.cpp -o minicpm
```

To utilise the q-bit quantized format, add the -q option to the command as follows:
```
For 8-bit format use:
./minicpm -q8

For 4-bit format use:
./minicpm -q4
```

To see all the available options, run
```
./minicpm --help
```
