# zephyr1_6b

**zephyr1_6b** is an implementation of [zephyr1_6b](https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b).
Supported formats are FP16 (3GBGB), 8-bit (1.6GB) and 4-bit (0.9GB).

## Install and Chat with zephyr1_6b.
```
git clone https://github.com/iangitonga/nanochatllms.cpp
cd nanochatllms.cpp/zephyr1_6b/
g++ -std=c++17 -I../ -O3 -fopenmp -ffast-math zephyr1_6b.cpp -o zephyr
./zephyr
```

If you have an Intel CPU that supports AVX and f16c compile with the following
 command to achieve higher performance:

```
g++ -std=c++17 -I../ -O3 -fopenmp -ffast-math -mavx -mf16c zephyr1_6b.cpp -o zephyr
```

To utilise the q-bit quantized format, add the -q option to the command as follows:
```
For 8-bit format use:
./zephyr -q8

For 4-bit format use:
./zephyr -q4
```

To see all the available options, run
```
./zephyr --help
```