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
import emcee
import corner
from multiprocessing import Pool
#from lightkurve import search_targetpixelfile
from scipy import signal
from astropy.table import Table, vstack

warnings.filterwarnings('ignore')


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
def ring_model(x, pdic, v=None):
    if v is not None:
        ##import pdb; pdb.set_trace()
        for i, param in enumerate(mcmc_params):
            #print(i, v[i])
            pdic[param] = v[i]

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


    #rp_rs, theta, phi, r_in, r_out = p
    #theta, phi = p
    #rp_rs, theta, phi = p
    #rp_rs, theta, phi, r_in = p
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
    start =time.time()
    model = ring_model(x, params.valuesdict())
    chi_square = np.sum(((data-model)/eps_data)**2)
    #print(params)
    print(chi_square)
    #print(np.max(((data-model)/eps_data)**2))

    return (data-model) / eps_data

#リングなしモデルをfitting
def no_ring_residual_transitfit(params, x, data, eps_data, names):
    global chi_square
    params_batman = set_params_batman(params, names)
    m = batman.TransitModel(params_batman, x)    #initializes model
    model = m.light_curve(params_batman)         #calculates light curve
    chi_square = np.sum(((data-model)/eps_data)**2)
    #print(params)
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

def clip_others_planet_transit(lc, duration, period, transit_time, others_duration, others_period, others_transit_time):
    for i in range(len(others_duration)):
        transit_start = others_transit_time[i] - (others_duration[i]/2)
        transit_end = others_transit_time[i] + (others_duration[i]/2)
        #import pdb; pdb.set_trace()
        lc = clip_transit_duration(lc, transit_start, transit_end)

        if others_transit_time[i] < np.median(lc['time'].value):
            while lc.time[-1].value < transit_start:
                others_transit_time[i] = others_transit_time[i] + others_period[i]
                transit_start = others_transit_time[i] - (others_duration[i]/2)
                transit_end = others_transit_time[i] + (others_duration[i]/2)
                lc = clip_transit_duration(lc, transit_start, transit_end)
        elif others_transit_time[i] > np.median(lc['time'].value):
            while transit_end < lc.time[0].value:
                others_transit_time[i] = others_transit_time[i] - others_period[i]
                transit_start = others_transit_time[i] - (others_duration[i]/2)
                transit_end = others_transit_time[i] + (others_duration[i]/2)
                lc = clip_transit_duration(lc, transit_start, transit_end)
    return lc


def clip_transit_duration(lc, transit_start, transit_end):
    if transit_end < lc.time[0].value:
        pass
    elif transit_start < lc.time[0].value and lc.time[0].value < transit_end:
        lc = lc[~(lc['time'].value < transit_start)]
    elif transit_start < lc.time[0].value and lc.time[-1].value < transit_end:
        #with open('huge_transit.csv') as f:
            #f.write()
        print('huge !')
        #記録する
    elif lc.time[0].value < transit_start and transit_end < lc.time[-1].value:
        lc = vstack([lc[lc['time'].value < transit_start], lc[lc['time'].value > transit_end]])
    elif lc.time[0].value < transit_start and lc.time[-1].value < transit_end:
        lc = lc[lc['time'].value < transit_start]
    elif lc.time[-1].value < transit_start:
        pass
    return lc

def preprocess_each_lc(lc, duration, period, transit_time):
    #planet_b_model = bls.get_transit_model(period=period, transit_time=transit_time, duration=duration)
    n_transit = int((lc.time[-1].value - transit_time) // period)
    transit_row_list = make_rowlist(n_transit, lc, transit_time, period)
    #minId = signal.argrelmin(lc.flux.value, order=3000)
    half_duration = (duration/2)*24*60
    twice_duration = (duration*2)*24*60 #durationを2倍、単位をday→mi
    lc_cut_point = half_duration + twice_duration
    lc_list=[]
    for i, transit in enumerate(transit_row_list):
        print('No.{} transit: '.format(i))

        start = int(transit - lc_cut_point)
        if start < 0:
            start = 0
        end = int(transit + lc_cut_point)
        if end > len(lc):
            end = len(lc)
        #each_lc = lc[start:end].normalize()
        each_lc = lc[start:end]
        print('before clip length: ', len(each_lc.flux))

        ###params setting
        noringnames = ["t0", "per", "rp", "a", "inc", "ecc", "w", "q1", "q2"]
        #values = [0.0, 4.0, 0.08, 8.0, 83.0, 0.0, 90.0, 0.2, 0.2]
        values = [transit_time+period*i, period, 0.08, 8.0, 83.0, 0.0, 90.0, 0.2, 0.2]
        mins = [-0.1, 4.0, 0.03, 4, 80, 0, 90, 0.0, 0.0]
        maxes = [0.1, 4.0, 0.2, 20, 110, 0, 90, 1.0, 1.0]
        #vary_flags = [True, False, True, True, True, False, False, True, True]
        vary_flags = [False, False, True, True, True, False, False, True, True]
        no_ring_params = set_params_lm(noringnames, values, mins, maxes, vary_flags)

        ###transit fitting and clip outliers
        while True:
            out = lmfit.minimize(no_ring_residual_transitfit,no_ring_params,args=(each_lc.normalize().time.value, each_lc.normalize().flux.value, each_lc.normalize().flux_err.value, noringnames),max_nfev=1000)
            flux_model = no_ring_model_transitfit_from_lmparams(out.params, each_lc.normalize().time.value, noringnames)
            clip_lc = each_lc.normalize().copy()
            clip_lc.flux = clip_lc.flux-flux_model
            _, mask = clip_lc.remove_outliers(return_mask=True)
            inverse_mask = np.logical_not(mask)
            if np.all(inverse_mask) == True:
                print('after clip length: ', len(each_lc.flux))
                each_lc.normalize().errorbar()
                plt.plot(each_lc.time.value, flux_model, label='fit_model')
                plt.legend()
                #plt.savefig('/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/fitting_result/figure/fitting_result_{}_{:.0f}.png'.format(datetime.datetime.now().strftime('%y%m%d%H%M'), chi_square), header=False, index=False)
                #plt.show()
                plt.close()
                break
            else:
                print('cliped:', len(each_lc.flux.value)-len(each_lc[~mask].flux.value))
                each_lc = each_lc[~mask]

        ###curve fiting

        each_lc_df = each_lc.to_pandas()
        before_transit = each_lc_df[each_lc_df.index < (transit_time+period*i)-(duration/2)]
        after_transit = each_lc_df[each_lc_df.index > (transit_time+period*i)+(duration/2)]
        out_transit = pd.concat([before_transit, after_transit])
        out_transit = out_transit.reset_index()
        out_transit = Table.from_pandas(out_transit)
        out_transit = lk.LightCurve(data=out_transit)
        model = lmfit.models.PolynomialModel()
        poly_params = model.make_params(c0=1, c1=0, c2=0, c3=0, c4=0, c5=0, c6=0, c7=0)
        result = model.fit(out_transit.flux.value, poly_params, x=out_transit.time.value)
        result.plot()
        plt.savefig('./folded_dfs/curvefit_figure/{}.png'.format('TOI' + str(item['TESS Object of Interest'])))
        #plt.show()
        plt.close()
        poly_model = np.polynomial.Polynomial([result.params.valuesdict()['c0'],\
                        result.params.valuesdict()['c1'],\
                        result.params.valuesdict()['c2'],\
                        result.params.valuesdict()['c3'],\
                        result.params.valuesdict()['c4'],\
                        result.params.valuesdict()['c5'],\
                        result.params.valuesdict()['c6'],\
                        result.params.valuesdict()['c7']])
        each_lc.flux = each_lc.flux.value/poly_model(each_lc.time.value)
        each_lc.flux_err = each_lc.flux_err.value/poly_model(each_lc.time.value)
        each_lc_df = each_lc.to_pandas()
        lc_list.append(each_lc_df)
    return lc_list

def folding_each_lc(lc_list):
    lc = pd.concat(lc_list)
    lc = lc.reset_index()
    lc = Table.from_pandas(lc)
    lc = lk.LightCurve(data=lc)
    lc = lc.normalize()
    print('total length: ', len(lc))
    return lc.fold(period=period, epoch_time=transit_time)

def getNearestRow(list, num):
    # リスト要素と対象値の差分を計算し最小値のインデックスを取得
    idx = np.abs(np.asarray(list) - num).argmin()
    return idx

def make_rowlist(n_transit, lc, transit_time, period):
    list = []
    if n_transit == 0:
        mid_transit_row = getNearestRow(lc.time.value, transit_time)
        list.append(mid_transit_row)
        return list
    else:
        for n in range(n_transit):
            target_val = transit_time + (period * n)
            mid_transit_row = getNearestRow(lc.time.value, target_val)
            list.append(mid_transit_row)
        return list

#if __name__ ==  '__main__':
'''
#使う行のみ抽出
sn_df=pd.read_csv('/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/toi-catalog.csv', encoding='shift_jis')
sn_df.columns=sn_df.iloc[3].values
sn_df=sn_df[4:]
#カラムを入れ替える。

sn_df=sn_df[['Signal-to-noise', 'Source Pipeline', 'TIC', 'Full TOI ID', 'TOI Disposition',
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

sn_df=sn_df[['Signal-to-noise', 'Source Pipeline', 'TIC', 'Full TOI ID', 'TOI Disposition']]
sn_df['Signal-to-noise'] = sn_df['Signal-to-noise'].fillna(0)
sn_df['Signal-to-noise'] = sn_df['Signal-to-noise'].astype(float)
sn_df = sn_df.sort_values('Signal-to-noise', ascending=False)

#SN比 ≧ 100のデータを抽出。
sn_df2 = sn_df[sn_df['Signal-to-noise'] >= 100]
sn_df2 = sn_df2.reset_index(drop=True)
#sn_df2 = sn_df2.drop(columns='index')
sn_df2.head()
sn_df2['TIC'] = sn_df2['TIC'].apply(lambda x:int(x))
'''

sn_df2 = pd.read_csv('/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/toi_overSN100.csv')
TIClist = sn_df2['TIC']
params_df = pd.read_excel('/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/TOI_parameters.xlsx')
#params_df[params_df['TESS Input Catalog ID']==TIClist[0]].T
#import pdb; pdb.set_trace()
for TIC in TIClist:
    param_df = params_df[params_df['TESS Input Catalog ID'] == TIC]

    tpf = lk.search_targetpixelfile('TIC {}'.format(TIC), mission='TESS', cadence="short").download()

    #tpf.plot(frame=100, scale='log', show_colorbar=True)
    try:
        lc = tpf.to_lightcurve(aperture_mask=tpf.pipeline_mask)
    except AttributeError:
        with open('error_tic.dat', 'a') as f:
            f.write(str(TIC) + '\n')
        continue
    for index, item in param_df.iterrows():
        duration = item['Planet Transit Duration Value [hours]'] / 24
        period = item['Planet Orbital Period Value [days]']
        transit_time = item['Planet Transit Midpoint Value [BJD]'] - 2457000.0 #translate BTJD
        others_duration = param_df[param_df.index!=index]['Planet Transit Duration Value [hours]'].values / 24
        others_period = param_df[param_df.index!=index]['Planet Orbital Period Value [days]'].values
        others_transit_time = param_df[param_df.index!=index]['Planet Transit Midpoint Value [BJD]'].values - 2457000.0 #translate BTJD
        #if lc.time.value[0] > transit_time or lc.time.value[-1] < transit_time:
        #    continue
        lc = clip_others_planet_transit(lc, duration, period, transit_time, others_duration, others_period, others_transit_time)
        lc_list = preprocess_each_lc(lc, duration, period, transit_time)
        try:
            folded_lc = folding_each_lc(lc_list)
        except ValueError:
            print('no transit!')
            continue
        folded_lc.errorbar()
        plt.savefig('./folded_dfs/folded_lc_figure/{}.png'.format('TOI' + str(item['TESS Object of Interest'])))
        #plt.show()
        plt.close()
        folded_lc.write('./folded_dfs/folded_lc_{}.csv'.format('TOI' + str(item['TESS Object of Interest'])))

    """
    lc_list = preprocess_each_lc(lc, duration, period, transit_time)
    folded_lc = folding_each_lc(lc_list)
    folded_lc.errorbar()
    plt.show()
    import pdb; pdb.set_trace()
    """

#get transit paramter from TESS database
'''
print('hello')
import pdb; pdb.set_trace()
period=2.20473541
transit_time=121.3585417
duration=0.162026

lc_list = preprocess_each_lc(lc, duration, period, transit_time)
folded_lc = folding_each_lc(lc_list)
folded_lc.errorbar()
plt.show()
#すべてのlightcurveの可視化
lc_collection = search_result.download_all()
lc_collection.plot();
plt.show()

#tpf.plot(frame=100, scale='log', show_colorbar=True)
lc = tpf.to_lightcurve(aperture_mask=tpf.pipeline_mask)
#lc.plot()
period = np.linspace(1, 3, 10000)
bls = lc.to_periodogram(method='bls', period=period, frequency_factor=500);


lc_list = preprocess_each_lc(lc, duration, period, transit_time)
folded_lc = folding_each_lc(lc_list)
folded_lc.errorbar()
#plt.show()
plt.close()
# foldingを任せた場合 すべての観測を平坦化しノーマライズする。これは"a stitched light curve"で表される。詳しくは Kepler data with Lightkurve.
#lc = lc_collection.stitch().flatten(window_length=901).remove_outliers()
#lc.plot();
#lc = lc_collection.stitch().flatten(window_length=901).remove_outliers()
### foldingを自分でやる場合
lc_single = search_result[3].download()
#lc_single.plot();
lc = lc_single.flatten(window_length=901).remove_outliers()
#lc.plot();
### orbital periodをBLSで探す
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
### カタログのorbital periodを使う
sn_df2[sn_df2['TIC']=='142087638']
#planet_b_period = float(sn_df2[sn_df2['TIC']=='142087638']['Orbital Period Value'].values[0])
#planet_b_t0 = float(sn_df2[sn_df2['TIC']=='142087638']['Epoch Value'].values[0])
#planet_b_dur = float(sn_df2[sn_df2['TIC']=='142087638']['Transit Duration Value'].values[0])
# Check the value for period
print('planet_b_period: ', planet_b_period)
print('planet_b_t0: ', planet_b_t0)
print('planet_b_dur: ', planet_b_dur)
### folding
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
