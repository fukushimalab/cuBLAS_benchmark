# 概要
cuBLASを用いたMatMulのパフォーマンスを測定します。  
各行列サイズ`(16, 32, 64, 128, 256, 512, 1024, 2048)`に対して任意の回数実行時間を測定して、中央値を出力します。  
`titan RTX`, `RTX3090`, `RTX4090`に対応しています。その他を追加する場合は`CMakeLists.txt`の`CMAKE_CUDA_ARCHITECTURES`に対応する値を追加してください。  

# Require
- CUDA
- g++
- CMake
- python
    - seaborn
    - matplotlib
    - pandas

# How To Use

```bash
mkdir build
cd build
cmake .. && make
./main
```

実行時引数を渡すことでloop回数を変更できます。
```bash
./main 1000
```

plot.pyでresultsからグラフを生成します。

自分用

コードを変更していない場合
```bash
cd build && make && ./main 100000 && cd .. && python3 plot.py
```

## Performance


### Linear-scale
#### titan
![](output_titan_linear.png)
#### 3090
![](output_3090_linear.png)
#### 4090
![](output_4090_linear.png)

### Log-scale
#### titan
![](output_titan_log.png)
#### 3090
![](output_3090_log.png)
#### 4090
![](output_4090_log.png)