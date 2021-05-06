## 使い方 (以下./python_extで作業する)

太陽系外惑星+リングのトランジットカーブを計算するコード。
元のコードはcで書かれているので、wrapperでpythonでも使えるようにしておく。

## 1. 下準備 
-  gcc, pythonなどを使えるようにしておく。  
- GNU Scientific Library (GSL)をcで使えるようにしておく。  
  Homebrewなどを使っているのならば、brew installなどでgslはインストールできる。  
  gslがインストールできたかどうかは、“-lgsl"などのリンカが通るかで確認しておく。  

## 2. ringのtransit light curveをpythonで使えるようにする
以下の二つのコマンドを(python_extフォルダで)打つと、shared library (c_compile_ring.so)が作成できる。  
ただし、  
/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.8/Headers  
はPython.hがある場所のpathに変更しておくようにする。  

- gcc -fpic -I/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.8/Headers -o c_compile_ring.o -c c_compile_ring.c  
- gcc -undefined dynamic_lookup -bundle -lgsl -lgslcblas c_compile_ring.o -o c_compile_ring.so  

## 3. 走らせる
2で作成したc_compile_ring.soをimportすれば使用できる。  
exoring_testフォルダに使用例(fit_test.py)を載せた。  
- python fit_test.py  

と打てば動く。このコードでは、実際にAizawa+2017で解析したデータ(Q8_l_KIC10403228_test_detrend.dat)を読み込み
それと指定したパラメータ (para_result_ring.datの2列目)でのモデルフラックスを並べてplotしてくれる。  

## 番外編 (./c_ext)
c_extフォルダでは実際にcを用いたフィッティングなどを行ってくれる。  
gslをインストールした後に、  
- gcc main.c mpfit.c -lgsl -lgslcblas  

として作成した実行ファイルを実行するとフィティングまでしてくれる。  
