# -*- coding: utf-8 -*-
"""ring_planet.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rGAI-83m5o2F0sgqzmlPsWz3Jq-M9VJb
"""

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
warnings.filterwarnings('ignore')



def name_load(filename):
    file_now = open(filename)
    lines = file_now.readlines()
    names = []
    for line in lines:
        itemList = line.split()
        names.append(itemList[0])
    return names

def make_dic(names, values):
    dic = {}
    for (i, name) in enumerate(names):
        dic[name] = values[i]
        print(name, values[i])
    return dic

def q_to_u_limb(q_arr):
    q1 = q_arr[0]
    q2 = q_arr[1]
    u1 = np.sqrt(q1) * 2 * q2
    u2 = np.sqrt(q1) * (1- 2 * q2)
    return np.array([u1, u2])

def set_params_batman(params_lm, names, limb_type ="quadratic"):

    params = batman.TransitParams()       #object to store transit parameters
    params.limb_dark =  limb_type        #limb darkening model
    q_arr = np.zeros(2)
    for i in range(len(names)):
        value = params_lm[names[i]]
        name = names[i]
        if name=="t0":
            params.t0 = value
        if name=="per":
            params.per = value
        if name=="rp":
            params.rp = value
        if name=="a":
            params.a = value
        if name=="inc":
            params.inc = value
        if name=="ecc":
            params.ecc = value
        if name=="w":
            params.w = value
        if name=="q1":
            q_arr[0] = value
        if name=="q2":
            q_arr[1] = value

    u_arr = q_to_u_limb(q_arr)
    params.u = u_arr
    return params

def set_params_lm(names, values, mins, maxes, vary_flags):
    params = lmfit.Parameters()
    for i in range(len(names)):
        if vary_flags[i]:
            params.add(names[i], value=values[i], min=mins[i], max = maxes[i], vary = vary_flags[i])
        else:
            params.add(names[i], value=values[i], vary = vary_flags[i])
    return params

# Ring model
# Input "x" (1d array), "pdic" (dic)
# Ouput flux (1d array)
def ring_model(x, pdic):
        q1, q2, t0, porb, rp_rs, a_rs, b, norm \
                = pdic['q1'], pdic['q2'], pdic['t0'], pdic['porb'], pdic['rp_rs'], pdic['a_rs'], pdic['b'], pdic['norm']
        theta, phi, tau, r_in, r_out \
                = pdic['theta'], pdic['phi'], pdic['tau'], pdic['r_in'], pdic['r_out']
        norm2, norm3 = pdic['norm2'], pdic['norm3']
        cosi = b/a_rs
        u = [2*np.sqrt(q1)*q2, np.sqrt(q1)*(1-2*q2)]
        u1, u2 = u[0], u[1]
        ecosw = pdic['ecosw']
        esinw = pdic['esinw']
        """ when e & w used: ecosw -> e, esinw -> w (deg)
        e = ecosw
        w = esinw*np.pi/180.0
        ecosw, esinw = e*np.cos(w), e*np.sin(w)
        """

        # see params_def.h
        pars = np.array([porb, t0, ecosw, esinw, b, a_rs, theta, phi, tau, r_in, r_out, rp_rs, q1, q2])
        times = np.array(x)
        return np.array(c_compile_ring.getflux(times, pars, len(times)))*(
                norm + norm2*(times-t0) + norm3*(times-t0)**2)


#リングありモデルをfitting
def ring_residual_transitfit(params, x, data, eps_data, names):
    global chi_square
    model = ring_model(x, params.valuesdict())
    chi_square = np.sum(((data-model)/eps_data)**2)
    print(params)
    print(chi_square)
    return (data-model) / eps_data

#リングありモデルをfitting
def no_ring_residual_transitfit(params, x, data, eps_data, names):
    global chi_square
    params_batman = set_params_batman(params, names)
    m = batman.TransitModel(params_batman, x)    #initializes model
    model = m.light_curve(params_batman)         #calculates light curve
    chi_square = np.sum(((data-model)/eps_data)**2)
    print(params)
    print(chi_square)
    return (data-model) / eps_data

def ring_model_transitfit_from_lmparams(params, x):
    model = ring_model(x, params.valuesdict())         #calculates light curve
    return model

def no_ring_model_transitfit_from_lmparams(params, x, names):
    params_batman = set_params_batman(params, names)
    m = batman.TransitModel(params_batman, x)    #initializes model
    model = m.light_curve(params_batman)
    return model

'''
"""使う行のみ抽出"""

df=pd.read_csv('/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/toi-catalog.csv', encoding='shift_jis')
df.columns=df.iloc[3].values
df=df[4:]

"""カラムを入れ替える。"""

df=df[['Signal-to-noise', 'Source Pipeline', 'TIC', 'Full TOI ID', 'TOI Disposition',
       'TIC Right Ascension', 'TIC Declination', 'TMag Value',
       'TMag Uncertainty', 'Epoch Value', 'Epoch Error',
       'Orbital Period Value', 'Orbital Period Error',
       'Transit Duration Value', 'Transit Duration Error',
       'Transit Depth Value', 'Transit Depth Error', 'Sectors',
       'Public Comment', 'Surface Gravity Value',
       'Surface Gravity Uncertainty', 'Signal ID', 'Star Radius Value',
       'Star Radius Error', 'Planet Radius Value', 'Planet Radius Error',
       'Planet Equilibrium Temperature (K) Value',
       'Effective Temperature Value', 'Effective Temperature Uncertainty',
       'Effective Stellar Flux Value', 'Centroid Offset',
       'TFOP Master', 'TFOP SG1a', 'TFOP SG1b', 'TFOP SG2', 'TFOP SG3',
       'TFOP SG4', 'TFOP SG5', 'Alerted', 'Updated']]

df['Signal-to-noise'] = df['Signal-to-noise'].fillna(0)
df['Signal-to-noise'] = df['Signal-to-noise'].astype(float)
df = df.sort_values('Signal-to-noise', ascending=False)
df

"""SN比 ≧ 100のデータを抽出。"""

df2 = df[df['Signal-to-noise'] >= 100]
df2 = df2.reset_index()
#df2 = df2.drop(columns='index')

df2.head()

search_result = lk.search_lightcurve('TIC 142087638', mission='TESS', exptime=120)

"""すべてのlightcurveの可視化"""

lc_collection = search_result.download_all()
#lc_collection.plot();

"""### foldingを任せた場合

すべての観測を平坦化しノーマライズする。これは"a stitched light curve"で表される。詳しくは Kepler data with Lightkurve.
"""

#lc = lc_collection.stitch().flatten(window_length=901).remove_outliers()
#lc.plot();

#lc = lc_collection.stitch().flatten(window_length=901).remove_outliers()

"""### foldingを自分でやる場合"""

lc_single = search_result[3].download()
#lc_single.plot();

lc = lc_single.flatten(window_length=901).remove_outliers()
#lc.plot();

"""### orbital periodをBLSで探す"""

# Create array of periods to search
period = np.linspace(1, 30, 10000)
# Create a BLSPeriodogram
bls = lc.to_periodogram(method='bls', period=period, frequency_factor=500);
#bls.plot();

planet_b_period = bls.period_at_max_power
planet_b_t0 = bls.transit_time_at_max_power
planet_b_dur = bls.duration_at_max_power

# Check the value for period
print('planet_b_period: ', planet_b_period)
print('planet_b_t0: ', planet_b_t0)
print('planet_b_dur: ', planet_b_dur)

"""### カタログのorbital periodを使う"""

df2[df2['TIC']=='142087638']

#planet_b_period = float(df2[df2['TIC']=='142087638']['Orbital Period Value'].values[0])
#planet_b_t0 = float(df2[df2['TIC']=='142087638']['Epoch Value'].values[0])
#planet_b_dur = float(df2[df2['TIC']=='142087638']['Transit Duration Value'].values[0])

# Check the value for period
print('planet_b_period: ', planet_b_period)
print('planet_b_t0: ', planet_b_t0)
print('planet_b_dur: ', planet_b_dur)

"""### folding"""

ax = lc.fold(period=planet_b_period, epoch_time=planet_b_t0).scatter()
#ax.set_xlim(-5, 5);
ax = lc.fold(period=planet_b_period, epoch_time=planet_b_t0).scatter()
#ax.set_xlim(-0.5, 0.5);

# Create a cadence mask using the BLS parameters
planet_b_mask = bls.get_transit_mask(period=planet_b_period,
                                     transit_time=planet_b_t0,
                                     duration=planet_b_dur)

masked_lc = lc[~planet_b_mask]
#ax = masked_lc.scatter();
#lc[planet_b_mask].scatter(ax=ax, c='r', label='Masked');

# Create a BLS model using the BLS parameters
planet_b_model = bls.get_transit_model(period=planet_b_period,
                                       transit_time=planet_b_t0,
                                       duration=planet_b_dur)

ax = lc.fold(planet_b_period, planet_b_t0).scatter()
planet_b_model.fold(planet_b_period, planet_b_t0).plot(ax=ax, c='r', lw=2)
#ax.set_xlim(-0.5, 0.5);
#ax.set_xlim(-5, 5);
#plt.show()
plt.cla()
plt.clf()




# read the data file (xdata, ydata, yerr)
xdata = lc.fold(planet_b_period, planet_b_t0).phase.value
ydata = lc.fold(planet_b_period, planet_b_t0).flux.value
yerr  = lc.fold(planet_b_period, planet_b_t0).flux_err.value
'''

#lm.minimizeのためのparamsのセッティング。これはリングありモデル
names = ["q1", "q2", "t0", "porb", "rp_rs", "a_rs",
         "b", "norm", "theta", "phi", "tau", "r_in",
         "r_out", "norm2", "norm3", "ecosw", "esinw"]
#values = [0.2, 0.2, 0.0, 4.0, (float(df2[df2['TIC']=='142087638']['Planet Radius Value'].values[0])*0.0091577) / float(df2[df2['TIC']=='142087638']['Star Radius Value'].values[0]), 40.0,
#          0.5, 1.0, 45.0, 45.0, 0.5, 1.5,
#          2.0/1.5, 0.0, 0.0, 0.0, 0.0]
values = [0.0, 0.7, 0.0, 4.0, 0.5, 10.7,
          1, 1, 30, 1, 1, 1.53,
          1.95, 0.0, 0.0, 0.0, 0.0]
#saturnlike_values = [0.2, 0.2, 0.0, 4.0, 0.08, 10.7,
#         1, 1, 26.7, 0, 1, 1.53,
#         1.95, 0.0, 0.0, 0.0, 0.0]
saturnlike_values = [0.0, 0.7, 0.0, 4.0, 0.18, 10.7,
          1, 1, 26.7, 0, 1, 1.53,
          1.95, 0.0, 0.0, 0.0, 0.0]
mins = [0.0, 0.0, -0.0001, 0.0, 0.0, 1.0,
        0.0, 0.9, 0.0, 0.0, 0.0, 1.0,
        1.1, -0.1, -0.1, 0.0, 0.0]
maxes = [1.0, 1.0, 0.0001, 100.0, 1.0, 1000.0,
         1.0, 1.1, 90.0, 90.0, 1.0, 7.0,
         10.0, 0.1, 0.1, 0.0, 0.0]
vary_flags = [False, False, False, False, True, False,
              False, False, True, True, False, False,
              False, False, False, False, False]
params = set_params_lm(names, values, mins, maxes, vary_flags)


t = np.linspace(-0.2, 0.2, 300)

#土星likeな惑星のパラメータで作成したモデル
saturnlike_params = set_params_lm(names, saturnlike_values, mins, maxes, vary_flags)
pdic_saturnlike = make_dic(names, saturnlike_values)
ymodel = ring_model(t, pdic_saturnlike)

#土星likeな惑星のパラメータで作成したlight curve
error_scale = 0.0001
eps_data = np.random.normal(size=t.size, scale=error_scale)
flux = ymodel + eps_data

"""
noringnames = ["t0", "per", "rp", "a", "inc", "ecc", "w", "q1", "q2"]
values = [0.0, 4.0, 0.08, 8.0, 83.0, 0.0, 90.0, 0.2, 0.2]
mins = [-0.1, 4.0, 0.03, 4, 80, 0, 90, 0.0, 0.0]
maxes = [0.1, 4.0, 0.2, 20, 110, 0, 90, 1.0, 1.0]
vary_flags = [True, False, True, True, True, False, False, True, True]
params = set_params_lm(noringnames, values, mins, maxes, vary_flags)
"""

import pdb; pdb.set_trace()

for i in range(1):
    out = lmfit.minimize(ring_residual_transitfit, params, args=(t, flux, error_scale, names), max_nfev=1000, method='nelder')

    #out = lmfit.minimize(no_ring_residual_transitfit, params, args=(t, flux, error_scale, noringnames))
    #flux_model = no_ring_model_transitfit_from_lmparams(out.params, t, noringnames)
    flux_model = ring_model_transitfit_from_lmparams(out.params, t)
    #import pdb; pdb.set_trace()
    plt.plot(t, flux, label='data')
    plt.plot(t, flux_model, label='fit_model')
    #plt.plot(t, ymodel, label='model')
    plt.legend()
    plt.savefig()
    #plt.show()
    #time.sleep(30)

    """csvに書き出し"""
    #input_df = pd.DataFrame.from_dict(params.valuesdict(), orient="index",columns=["input_value"])
    input_df = pd.DataFrame.from_dict(saturnlike_params.valuesdict(), orient="index",columns=["input_value"])
    output_df = pd.DataFrame.from_dict(out.params.valuesdict(), orient="index",columns=["output_value"])
    input_df=input_df.applymap(lambda x: '{:.6f}'.format(x))
    output_df=output_df.applymap(lambda x: '{:.6f}'.format(x))
    #df = input_df.join((output_df, pd.Series(vary_flags, index=noringnames, name='vary_flags')))
    df = input_df.join((output_df, pd.Series(vary_flags, index=names, name='vary_flags')))
    df.to_csv('/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/fitting_result/fitting_result_{}_{:.0f}.csv'.format(datetime.datetime.now().strftime('%y%m%d%H%M'), chi_square), header=False, index=False)
    #import pdb; pdb.set_trace()


"""
parfile  = "/Users/u_tsubasa/work/ring_planet_research/ring_transit/python_ext/exoring_test/test/para_result_ring.dat"
parvalues = []
file = open(parfile, "r")
lines = file.readlines()
for (i, line) in enumerate(lines):

    itemList = line.split()
    parvalues.append(float(itemList[1]))


parvalues = np.array(parvalues)
"""

"""
names = ["t0", "per", "rp", "a", "inc", "ecc", "w", "q1", "q2"]
values = [0, 3.5, 0.08, 8, 83, 0, 90, 0.2, 0.2]
mins = [0, 3.5, 0.03, 4, 80, 0, 90, 0.0, 0.0]
maxes = [0, 3.5, 0.2, 20, 110, 0, 90, 1.0, 1.0]
vary_flags = [False, False, True, True, True, False, False, True, True]
pdic = make_dic(names, values)
import pdb; pdb.set_trace()
out = lmfit.minimize(ring_residual_transitfit, params, args=(xdata, ydata, yerr, names))
flux_model = model_transitfit_from_lmparams(out.params, xdata, names)
print(lmfit.fit_report(out.params))


#print test data
#pdic = make_dic(names, parvalues)
#plt.plot(xdata, flux_model, label='model')
#plt.plot(xdata, ydata, label='data')
"""
