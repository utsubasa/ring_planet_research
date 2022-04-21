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
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def calc_tidal_tau():
    q=10**6.5 #Qp: the tidal dissipation function of the planet
    c=0.25 #C:the dimensionless moment of inertia of the planet
    kp=1.5 #Love number
    return (2*c*q/(3*kp)) * (Mp/Ms) * pow((a/Rp),3) * (porb/(2*np.pi))


#if __name__ ==  '__main__':
homedir = '/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani'

oridf = pd.read_csv(f'{homedir}/exofop_tess_tois.csv')
df = oridf[oridf['Planet SNR']>100]
df = df.sort_values('Planet SNR', ascending=False)
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
    files = os.listdir(f'{homedir}/TOI{TOI}')
    files.remove('.DS_Store')
    file = sorted(files, key=natural_keys)[0]
    p_data = pd.read_csv(f'{homedir}/TOI{TOI}/{file}')
    #a、period,Rpの値を使用。a,rpはRsの値を使って単位をkmにする
    porb = p_data.iat[3,1]/365 #porb -> yearに変換
    Rp = p_data.iat[4,1]
    a = p_data.iat[5,1]
    Ms = TOIdf['Stellar Mass (M_Sun)'][0]*0.0009546
    #Mpはカタログ値
    try:
        Mp = TOIdf['pl_massj'][0]
    except:
        import pdb; pdb.set_trace()

    tau = calc_tidal_tau()
    import pdb; pdb.set_trace()
#結果をstellar ageと比較。

import pdb; pdb.set_trace()
