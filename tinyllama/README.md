# tinyllama

**tinyllama** is an implementation of [TinyLlama-1.1B-Chat-v0.4](https://github.com/jzhang38/TinyLlama).
Supported formats are FP16 (2.2GB), 8-bit (1.2GB) and 4-bit (0.62GB).

## Install and Chat with Tinyllama.
```
git clone https://github.com/iangitonga/nanochatllms.cpp
cd nanochatllms.cpp/tinyllama/
g++ -std=c++17 -I../ -O3 -fopenmp -ffast-math tinyllama.cpp -o tinyllama
./tinyllama
```

If you have an Intel CPU that supports AVX and f16c compile with the following
 command to achieve higher performance:

```
g++ -std=c++17 -I../ -O3 -fopenmp -ffast-math -mavx -mf16c tinyllama.cpp -o tinyllama
```

To utilise the q-bit quantized format, add the -q option to the command as follows:
```
For 8-bit format use:
./tinyllama -q8

For 4-bit format use:
./tinyllama -q4
```

To see all the available options, run
```
./tinyllama --help
```
