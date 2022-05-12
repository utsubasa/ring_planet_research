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
import re

warnings.filterwarnings('ignore')

def atoi(text):
    """ソートがうまくいくようにする関数"""
    return int(text) if text.isdigit() else text

def natural_keys(text):
    """ソートがうまくいくようにする関数"""
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def calc_damp_tau(p_data,df):
    """calculate the damping timescale of planetary spin axis

    Parameters:
        qp: Qp. the tidal dissipation function of the planet
        c:the dimensionless moment of inertia of the planet
        kp: Love number

    """
    #a、period,Rpの値を使用。a,rpはRsの値を使って単位をkmにする
    q=10**6.5 #Qp: the tidal dissipation function of the planet
    c=0.25 #C:the dimensionless moment of inertia of the planet
    kp=1.5 #Love number

    Teff = df['Stellar Teff (K)'].iloc[-1]
    Rs_cm = df['Stellar Radius (R_Sun)'].iloc[-1]* 6.9634e+10 #[cm]
    Rs_earth = df['Stellar Radius (R_Sun)'].iloc[-1]* 109 #[R_Earth]
    Ms = df['Stellar Mass (M_Sun)'].iloc[-1]
    Ms_kg = Ms*1.9891*np.power(10.0, 30) #[kg]

    porb = p_data.iat[3,1]/365 #porb -> yearに変換
    rp_rs = p_data.iat[4,1]
    rp_cm = rp_rs*Rs_cm
    a_rs = p_data.iat[5,1]
    a_cm = a_rs*Rs_cm

    '''カタログ値の惑星質量を使う場合'''
    Mp = df['pl_masse'].iloc[-1]*5.972e+24 #カタログ値。[kg]
    #import pdb; pdb.set_trace()
    if np.isnan(Mp):
        '''カタログ値の惑星質量を使わない場合'''
        sigma_sb = 5.67e-5
        F_inc = (sigma_sb * (Teff**4) * (Rs_cm**2)) / (4* np.pi * (a_cm**2))
        Rp_earth = rp_cm/6378e+5 #[R_earth]
        Mp = (0.337*np.power(Rp_earth, 1/0.53)*np.power(F_inc, 0.03/0.53))*5.972e+24 #[kg]

    return (2*c*q/(3*kp)) * (Mp/Ms_kg) * np.power((a_rs/rp_rs),3) * (porb/(2*np.pi))

#if __name__ ==  '__main__':
homedir = '/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani'
no_data_list = [4726.01,372.01,352.01,2617.01,2766.01,2969.01,2989.01,2619.01,2626.01,2624.01,2625.01,
                2622.01,3041.01,2889.01,4543.01,3010.01,2612.01,4463.01,4398.01,4283.01,4145.01,3883.01,
                4153.01,3910.01,3604.01,3972.01,3589.01,3735.01,4079.01,4137.01,3109.01,3321.01,3136.01,
                3329.01,2826.01,2840.01,3241.01,2724.01,2704.01,2803.01,2799.01,2690.01,2745.01,2645.01,
                2616.01,2591.01,2580.01,2346.01,2236.01,2047.01,2109.01,2031.01,2040.01,2046.01,1905.01,
                2055.01,1518.01,1567.01,1355.01,1498.01,1373.01,628.01,1482.01,1580.01,1388.01,1310.01,
                1521.01,1123.01] #short がないか、SPOCがないか
no_perioddata_list = [1134.01,1897.01,2423.01,2666.01,4465.01]#exofopの表にperiodの記載無し。1567.01,1656.01もperiodなかったがこちらはcadence=’short’のデータなし。
no_signal_list = [2218.01,212.01,1823.01] #トランジットのsignalが無いか、ノイズに埋もれて見えない

oridf = pd.read_csv(f'{homedir}/exofop_tess_tois.csv')
df = oridf[oridf['Planet SNR']>100]
df = df.sort_values('Planet SNR', ascending=False)
df = df.set_index(['TOI'])
df = df.drop(index=no_data_list, errors='ignore')
df = df.drop(index=no_perioddata_list, errors='ignore')
df = df.drop(index=no_signal_list, errors='ignore')
df = df.reset_index()
df['TOI'] = df['TOI'].astype(str)
TOIlist = df['TOI']

nasa_df = pd.read_csv('/Users/u_tsubasa/Downloads/PS_2022.04.14_04.34.07.csv')
nasa_df = nasa_df[nasa_df.index < 124]
nasa_df['TIC ID'] = nasa_df['tic_id'].apply(lambda x: x[4:])
nasa_df['TIC ID'] = nasa_df['TIC ID'].astype(int)
df['log Period'] = np.log10(df['Period (days)'])
df = pd.merge(df, nasa_df, how='left')
df['log Mass'] = np.log10(df['pl_masse'])

#lmfit(mcmc)のフィッティングパラメータを読み込む。
homedir='/Users/u_tsubasa/Dropbox/ring_planet_research/fitting_result/data'

taulist = []
for TOI in TOIlist:
    TOIdf = df[df.TOI==TOI]
    try:
        files = os.listdir(f'{homedir}/TOI{TOI}')
    except FileNotFoundError:
        df = df.set_index(['TOI'])
        df = df.drop(TOI)
        df = df.reset_index()
        continue
    try:
        files.remove('.DS_Store')
    except ValueError:
        pass
    file = sorted(files, key=natural_keys)[0]
    p_data = pd.read_csv(f'{homedir}/TOI{TOI}/{file}')
    tau = calc_damp_tau(p_data,TOIdf)
    taulist.append(tau)
#結果をstellar ageと比較。
df['tdamp']=taulist
df[~df['st_age'].isnull()][['TOI','st_age','tdamp']]
import pdb; pdb.set_trace()
mcmc_list=['267.01','585.01','615.01','624.01','665.01','857.01','1025.01','1092.01','1283.01','1292.01','1431.01','1924.01','1963.01','1976.01','2020.01','2140.01','3460.01','4606.01','1963.01']
df.loc[mcmc_list, :][['TOI','st_age','tdamp', 'pl_orbsmax']]