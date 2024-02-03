# zephyr1_6b

**zephyr1_6b** is an implementation of [zephyr1_6b](https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b).
Supported formats are FP16 (3GB), 8-bit (1.6GB) and 4-bit (0.9GB).

## Install and Chat with zephyr1_6b.
```
git clone https://github.com/iangitonga/nanochatllms.cpp
cd nanochatllms.cpp/zephyr1_6b/
g++ -std=c++17 -I../ -O3 -fopenmp zephyr1_6b.cpp -o zephyr
./zephyr
```

If you have an Intel CPU that supports AVX and f16c compile with the following
 command to achieve higher performance:

```
g++ -std=c++17 -I../ -O3 -fopenmp -mavx -mf16c zephyr1_6b.cpp -o zephyr
```

To see all the available options, run
```
./zephyr --help
```