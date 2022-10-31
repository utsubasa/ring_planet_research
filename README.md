# ring_planet_research
## 想定しているディレクトリ構造
```
.
├──emcee_fit.py
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
        
├──new_lmfit_result
    ├──fit_p_data
        ├──no_ring_model
            ├──TOI170.01
                ├──TOI170.01_0.87.csv
            ・・・
        └──ring_model
            ├──TOI170.01
                ├──TOI170.01_-26_0.csv
                ・・・
            ・・・
          
    ├──transit_fit
        ├──TOI170.01
            ├──TOI170.01_-26_0.pdf
            ・・・
        ・・・
    ├──fit_report
        ├──TOI101.01.txt
        ・・・
        
    └──illustration
        ├──TOI170.01
            ├──TOI170.01_-26_0.pdf
            ・・・
        ・・・
```
