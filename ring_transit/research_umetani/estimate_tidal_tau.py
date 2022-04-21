# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lmfit
import lightkurve as lk
from fit_model import model
import warnings
import c_compile_ring
import batman
import datetime
from decimal import *
import time
#from lightkurve import search_targetpixelfile
from scipy.stats import t, linregress
from astropy.table import Table, vstack
import astropy.units as u
from decimal import Decimal, ROUND_HALF_UP
import os
import celerite
from celerite import terms
from scipy.optimize import minimize

warnings.filterwarnings('ignore')



#if __name__ ==  '__main__':
homedir = '/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani'

oridf = pd.read_csv(f'{homedir}/exofop_tess_tois.csv')
df = oridf[oridf['Planet SNR']>100]
each_lc_anomalylist = [102.01,106.01,114.01,123.01,135.01,150.01,163.01,173.01,349.01,471.01,
                        505.01,625.01,626.01,677.01,738.01,834.01,842.01,845.01,858.01,
                        934.01,987.01,1019.01,1161.01,1163.01,1176.01,1259.01,1264.01,
                        1274.01,1341.01,1845.01,1861.01,1924.01,1970.01,2000.01,2014.01,2021.01,2131.01,
                        2200.01,2222.01,3846.01,4087.01,4486.01]#トランジットが途中で切れてcurvefitによって変なかたちになっているeach_lcを削除したデータ
mtt_shiftlist = [129.01,199.01,236.01,758.01,774.01,780.01,822.01,834.01,1050.01,1151.01,1165.01,1236.01,1265.01,
                1270.01,1292.01,1341.01,1721.01,1963.01,2131.01] #mid_transit_time shift　
no_data_list = [4726.01,372.01,352.01,2617.01,2766.01,2969.01,2989.01,2619.01,2626.01,2624.01,2625.01,
                2622.01,3041.01,2889.01,4543.01,3010.01,2612.01,4463.01,4398.01,4283.01,4145.01,3883.01,
                4153.01,3910.01,3604.01,3972.01,3589.01,3735.01,4079.01,4137.01,3109.01,3321.01,3136.01,
                3329.01,2826.01,2840.01,3241.01,2724.01,2704.01,2803.01,2799.01,2690.01,2745.01,2645.01,
                2616.01,2591.01,2580.01,2346.01,2236.01,2047.01,2109.01,2031.01,2040.01,2046.01,1905.01,
                2055.01,1518.01,1567.01,1355.01,1498.01,1373.01,628.01,1482.01,1580.01,1388.01,1310.01,
                1521.01,1123.01] #short がないか、SPOCがないか
no_perioddata_list = [1134.01,1897.01,2423.01,2666.01,4465.01]#exofopの表にperiodの記載無し。1567.01,1656.01もperiodなかったがこちらはcadence=’short’のデータなし。
no_signal_list = [2218.01,212.01,1823.01] #トランジットのsignalが無いか、ノイズに埋もれて見えない

#done_list = [4470.01,495.01,423.01,398.01,165.01,1148.01,157.01,1682.01,1612.01,112.01,656.01]
done_list = os.listdir('/Users/u_tsubasa/Dropbox/ring_planet_research/folded_lc/figure/catalog_v')
df = df.set_index(['TOI'])
#df = df.drop(index=each_lc_anomalylist)
#df = df.drop(index=mtt_shiftlist, errors='ignore')
#df = df.drop(index=done_list, errors='ignore')
df = df.drop(index=no_data_list, errors='ignore')
df = df.drop(index=no_signal_list, errors='ignore')
df = df.reset_index()

df = df.sort_values('Planet SNR', ascending=False)
df['TOI'] = df['TOI'].astype(str)
TOIlist = df['TOI']
'''
plt.scatter(df['Period (days)'],df['Planet Radius (R_Earth)'], color='k')
plt.xscale('log')
plt.xlabel('Period[days]')
plt.ylabel('Radius[Earth Radii]')
plt.show()
plt.scatter(df['Period (days)'],df['Planet Eq Temp (K)'], color='k')
plt.xscale('log')
plt.xlabel('Period[days]')
plt.ylabel('Planet Eq Temp (K)')
plt.show()
'''
nasa_df = pd.read_csv('/Users/u_tsubasa/Downloads/PS_2022.04.14_04.34.07.csv')
nasa_df = nasa_df[nasa_df.index < 124]
nasa_df['TIC ID'] = nasa_df['tic_id'].apply(lambda x: x[4:])
nasa_df['TIC ID'] = nasa_df['TIC ID'].astype(int)
df['log Period'] = np.log10(df['Period (days)'])
df = df.merge(nasa_df, on='TIC ID')
df['log Mass'] = np.log10(df['pl_masse'])

#lmfit(mcmc)のフィッティングパラメータを読み込む。
homedir='/Users/u_tsubasa/Dropbox/ring_planet_research/fitting_result/data'


#a、period,Rpの値を使用。aばRsの値を使って単位をAUに戻す

#以下に数式を記述

#結果をstellar ageと比較。

import pdb; pdb.set_trace()
