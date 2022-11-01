# ring_planet_research
## emcee_fit.pyの使い方
### Requirement
- 以下のパッケージを使用しますので、事前にpipやconda等でインストールする必要があります。
```
astropy==4.2.1
corner==2.2.1
emcee==3.0.2
h5py=2.10.0
lmfit==1.0.2
matplotlib==3.3.4
numpy==1.20.1
lightkurve==2.0.10
pandas==1.2.4
```
バージョンの互換性については未調査事項です。

- [https://github.com/2ndmk2/ring_transit](https://github.com/2ndmk2/ring_transit)で使用されている`c_compile_ring.so`も必要です。
こちらのファイルはemcee_fit.pyと同じディレクトリに置いてください。
- 惑星のパラメータを取得するためにexofop_tess_tois_2022-09-13.csvが必要です。こちらもemcee_fit.pyと同じディレクトリに置いてください。

### 想定しているディレクトリ構造
以下のディレクトリ構造でemcee_fit.pyが実行されることを想定しています。
```
./
├──c_compile_ring.so
├──emcee_fit.py
├──exofop_tess_tois_2022-09-13.csv
├──folded_lc_data
    ├──20220913
        ├──obs_t0
            └──csv
                ├──TOI170.01.csv
                ...
                
    ├──obs_t0
        ├──TOI101.01.csv
        ・・・
        
├──lmfit_result
    ├──fit_p_data
        ├──no_ring_model
            ├──TOI101.01
                ├──TOI101.01_0.86.csv
            ・・・
        └──ring_model
            ├──TOI101.01
                ├──TOI101.01_-19_8.csv
                ・・・
            ・・・
          
    ├──transit_fit
        ├──TOI101.01
            ├──TOI101.01_-19_8.pdf
            ・・・
        ・・・
    ├──fit_report
        ├──TOI101.01.txt
        ・・・
        
    └──illustration
        ├──TOI101.01
            ├──TOI101.01_-19_8.pdf
            ・・・
        ・・・

```
### Usage
以下のようにコマンドライン引数を指定してemcee_fit.pyを実行することで、指定した条件でmcmcの結果を表すコーナープロット、トランジットフィット、walkersのステップ、トランジットを起こす惑星のポンチ絵の図と、それぞれのwalkersのフィッティングパラメータが記録されます。

`python emcee_fit.py free_angle TOI183.01`

１つめのコマンドライン引数`free_angle`はリングの向きに関する設定で、free_angle,edge_onの2種類を入力することができます。free_angleでは惑星リングの向きを固定せずに事後分布の推定を行います。edge_onでは惑星リングの向きが惑星軌道と並行になるようにリングを固定して事後分布の推定を行います。

2つ目のコマンドライン引数`TOI183.01`ではmcmcを実行する天体を指定します。lmfit_result/fit_p_data/ring_modelにデータが存在する天体のみ指定することが可能です。
