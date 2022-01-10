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
import exoplanet as xo
import pymc3 as pm
import theano.tensor as tt
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
    #print(chi_square)
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
    #print(chi_square)
    return (data-model) / eps_data

def ring_model_transitfit_from_lmparams(params, x):
    model = ring_model(x, params.valuesdict())         #calculates light curve
    return model

def no_ring_model_transitfit_from_lmparams(params, x, names):
    params_batman = set_params_batman(params, names)
    m = batman.TransitModel(params_batman, x)    #initializes model
    model = m.light_curve(params_batman)
    return model

def clip_transit_hoge(lc, duration, period, transit_time, clip_transit=False):
    contain_transit = 0
    transit_time_list = []
    #print(f'transit_time: {transit_time}')
    transit_start = transit_time - (duration/2)
    transit_end = transit_time + (duration/2)

    if transit_time < np.median(lc['time'].value):
        while judge_transit_contain(lc, transit_start, transit_end) < 6:
            #print(f'transit_time: {transit_time}')
            transit_start = transit_time - (duration/2)
            transit_end = transit_time + (duration/2)
            case = judge_transit_contain(lc, transit_start, transit_end)
            #print('case:', case)
            #他の惑星処理に使う場合は以下の処理を行う。
            if len(lc[(lc['time'].value > transit_start) & (lc['time'].value < transit_end)]) != 0:
                if clip_transit == True:
                    lc = remove_transit_signal(case, lc, transit_start, transit_end)
                else:
                    if case == 2 or case == 3 or case == 4 or case == 5:
                        contain_transit = 1
                        transit_time_list.append(transit_time)
                    else:
                        pass
            else:
                pass
                #print("don't need clip because no data around transit")
            transit_time = transit_time + period
    elif transit_time > np.median(lc['time'].value):
        while judge_transit_contain(lc, transit_start, transit_end) > 1:
            #print(f'transit_time: {transit_time}')
            transit_start = transit_time - (duration/2)
            transit_end = transit_time + (duration/2)
            case = judge_transit_contain(lc, transit_start, transit_end)
            #print('case:', case)
            if len(lc[(lc['time'].value > transit_start) & (lc['time'].value < transit_end)]) != 0:
                if clip_transit == True:

                    lc = remove_transit_signal(case, lc, transit_start, transit_end)
                else:
                    if case == 2 or case == 3 or case == 4 or case == 5:
                        contain_transit = 1
                        transit_time_list.append(transit_time)
                    else:
                        pass
            else:
                print("don't need clip because no data around transit")
            transit_time = transit_time - period

    return lc, contain_transit, transit_time_list


def judge_transit_contain(lc, transit_start, transit_end):
    if transit_end < lc.time[0].value: # || -----
        case = 1
    elif transit_start < lc.time[0].value and lc.time[0].value < transit_end and transit_end < lc.time[-1].value: # | --|---
        case = 2
    elif transit_start < lc.time[0].value and lc.time[-1].value < transit_end: # | ----- |
        case = 3
    elif lc.time[0].value < transit_start and transit_start < lc.time[-1].value and lc.time[0].value < transit_end and transit_end < lc.time[-1].value: # --|-|--
        case = 4
    elif lc.time[0].value < transit_start and transit_start < lc.time[-1].value and lc.time[-1].value < transit_end: # ---|-- |
        case = 5
    elif lc.time[-1].value < transit_start: # ----- ||
        case = 6
    else:
        print('unexcepted case')
        import pdb; pdb.set_trace()
    return case


def remove_transit_signal(case, lc, transit_start, transit_end):
    if case == 1: # || -----
        pass
    elif case == 2: # | --|---
        lc = lc[~(lc['time'].value < transit_start)]
    elif case == 3: # | ----- |
        #with open('huge_transit.csv') as f:
            #f.write()
        print('huge !')
        #記録する
    elif case == 4: # --|-|--
        #lc = vstack([lc[lc['time'].value < transit_start], lc[lc['time'].value > transit_end]])
        lc = lc[(lc['time'].value < transit_start) | (lc['time'].value > transit_end)]

    elif case == 5: # ---|-- |
        lc = lc[lc['time'].value < transit_start]
    elif case == 6: # ----- ||
        pass

    return lc

def judge_outliers(array):
    # 2. Determine mean and standard deviation
    mean = np.mean(array)
    std_dev = np.std(array)
    # 3. Normalize array around 0
    zero_based = abs(array - mean)
    # 4. Define maximum number of standard deviations
    max_deviations = 2
    # 5. Access only non-outliers using Boolean Indexing
    outliers = array[~(zero_based < max_deviations * std_dev)]
    print(outliers)
    if len(outliers) !=0:
        return True
    else:
        return False

def detect_transit_epoch(folded_lc, transit_time, period):
    """トランジットエポックの検出"""
    #epoch_all_time = ( (folded_lc.time_original.value - transit_time) + 0.5*period ) / period
    epoch_all_time = ( (folded_lc.time_original.value - transit_time)) / period
    #epoch_all= np.array(epoch_all_time, dtype = int)
    epoch_all = [Decimal(str(x)).quantize(Decimal('0'), rounding=ROUND_HALF_UP).to_eng_string() for x in epoch_all_time]
    epoch_all = np.where(epoch_all == -0, 0, epoch_all).astype('int16')
    epoch_all_list = np.unique(epoch_all)
    epoch_all_list = np.sort(epoch_all_list)
    folded_lc.epoch_all = epoch_all
    if len(np.unique(epoch_all)) != len(epoch_all_list):
        print('check: len(np.unique(epoch_all)) != len(epoch_all_list).')
        import pdb; pdb.set_trace()
    else:
        pass
    return folded_lc, epoch_all_list

def transit_params_setting(rp_rs, period):
    global names
    """トランジットフィッティングパラメータの設定"""
    names = ["t0", "per", "rp", "a", "inc", "ecc", "w", "q1", "q2"]
    if np.isnan(rp_rs):
        values = [0, period, 0.02, 10, 87, 0, 90, 0.3, 0.2]
    else:
        values = [0, period, rp_rs, 10, 87, 0, 90, 0.3, 0.2]
    mins = [-0.7, period*0.6, 0.001, 0.1, 70, 0, 90, 0.0, 0.0]
    maxes = [0.7, period*1.9, 1.0, 100, 110, 0, 90, 1.0, 1.0]
    vary_flags = [True, False, True, True, False, False, False, False, False]
    return set_params_lm(names, values, mins, maxes, vary_flags)

def transit_case_is4(each_lc, duration, period, flag=False):
    """（period推定時で、）トランジットの時間帯丸々がデータに収まっていない場合は解析中断"""
    transit_start = -duration/period
    transit_end = duration/period
    #if estimate_period == True and judge_transit_contain(each_lc, transit_start, transit_end) != 4:
    if judge_transit_contain(each_lc, transit_start, transit_end) == 4:
        flag = True
    else:
        flag = False
    return flag

def aroud_midtransitdata_isexist(each_lc, flag=False):
    """midtransitのデータ点がlightcurveにない場合は解析中断"""
    if len(each_lc[(each_lc.time < 0.01) & (each_lc.time > -0.01)]) == 0:
        each_lc.errorbar()
        plt.title('no data in mid transit')
        plt.savefig(f'{homedir}/fitting_result/figure/error_lc/{KOInumber}_{str(i)}.png', header=False, index=False)
        plt.close()
        flag = False
    else:
        flag = True
    return flag

def nospace_in_transit(each_lc, transit_start, transit_end, flag=False):
    """トランジット中に空白の期間があったら解析中断するための関数"""
    delta = each_lc.time.value[:-1]-each_lc.time.value[1:]
    duration_flag = ((each_lc.time > transit_start*1.1) & (each_lc.time < transit_end*1.1))[:-1]
    delta = delta[duration_flag]
    ###deltaの全ての要素が同じ→空白がない
    if np.all(delta) == True:
        flag = True
    ###2シグマ以上の外れ値がある→データの空白がある
    elif judge_outliers(delta) == True:
        #plt.scatter()
        flag = False
    else:
        flag = True
    return flag

def transit_fit_and_remove_outliers(lc, t0dict, outliers, estimate_period=False, lc_type=None):
    #不具合の出るlcはcurvefittingでも弾けるようにno_use_lcを定義
    no_use_lc = False
    while True:
        """transit fitting"""
        try:
            flag_time = np.abs(lc.time.value)<1.0
            lc = lc[flag_time]
            time = lc.time.value
            flux = lc.flux.value
            flux_err = lc.flux_err.value
            out = lmfit.minimize(no_ring_residual_transitfit, params, args=(time, flux, flux_err, names), max_nfev=10000)
            #print(lmfit.fit_report(out))
        except TypeError:
            print('TypeError: out')
            import pdb; pdb.set_trace()
            continue
        except ValueError:
            print('cant fiting')

            import pdb; pdb.set_trace()
            #return lc, outliers, out, t0dict, no_use_lc
            continue

        """remove outliers"""
        if lc_type == 'each':
            try:
                if np.isfinite(out.params["t0"].stderr):
                    #print(out.params.pretty_print())
                    #time_now_arr.append(0.5 * np.min(each_lc.time_original.value) + 0.5* np.max(each_lc.time_original.value))
                    flux_model = no_ring_model_transitfit_from_lmparams(out.params, time, names)
                    clip_lc = lc.copy()
                    clip_lc.flux = np.sqrt(np.square(flux_model - clip_lc.flux))
                    _, mask = clip_lc.remove_outliers(return_mask=True)
                    inverse_mask = np.logical_not(mask)

                    if np.all(inverse_mask) == True:
                        #print(f'after clip length: {len(each_lc.flux)}')
                        if estimate_period == False:
                            fig = plt.figure()
                            ax1 = fig.add_subplot(2,1,1) #for plotting transit model and data
                            ax2 = fig.add_subplot(2,1,2) #for plotting residuals
                            lc.errorbar(ax=ax1, color='black', marker='.')
                            ax1.plot(time,flux_model, label='fit_model', color='red')
                            try:
                                outliers = vstack(outliers)
                                outliers.errorbar(ax=ax1, color='cyan', label='outliers(each_lc)', marker='.')
                            except ValueError:
                                pass
                            ax1.legend()
                            ax1.set_title(f'chi square/dof: {int(chi_square)}/{len(lc)} ')
                            residuals = lc - flux_model
                            residuals.errorbar(ax=ax2, color='black', marker='.')
                            ax2.plot(time,np.zeros(len(time)), label='fitting model', color='red')
                            ax2.set_ylabel('residuals')
                            os.makedirs(f'{homedir}/fitting_result/figure/each_lc/{KOInumber}', exist_ok=True)
                            plt.tight_layout()
                            plt.savefig(f'{homedir}/fitting_result/figure/each_lc/{KOInumber}/{KOInumber}_{str(i)}.png', header=False, index=False)
                            #ax.set_xlim(-1, 1)
                            #plt.show()
                            plt.close()
                        else:
                            #pass

                            t0dict[i] = [transit_time+(period*i)+out.params["t0"].value, out.params["t0"].stderr]
                            #t0dict[i] = [out.params["t0"].value, out.params["t0"].stderr]
                            #each_lc = clip_lc
                        break
                    else:
                        #print('removed bins:', len(each_lc[mask]))
                        outliers.append(lc[mask])
                        lc = lc[~mask]

            except TypeError:
                each_lc.errorbar()
                plt.xlim(-1, 1)
                plt.title('np.isfinite(out.params["t0"].stderr)==False')
                plt.savefig(f'{homedir}/fitting_result/figure/error_lc/{KOInumber}_{str(i)}.png', header=False, index=False)
                plt.close()
                print(lmfit.fit_report(out))
                no_use_lc = True
                #import pdb; pdb.set_trace()
                break
        else:
            break
    return lc, outliers, out, t0dict, no_use_lc

def estimate_period(t0dict, period):
    """return estimated period or cleaned light curve"""
    t0df = pd.DataFrame.from_dict(t0dict, orient='index', columns=['t0', 't0err'])
    x = t0df.index.values
    y = t0df['t0']
    yerr = t0df['t0err']
    try:
        res = linregress(x, y)
    except ValueError:
        print('ValueError: Inputs must not be empty.')
        import pdb; pdb.set_trace()
    estimated_period = res.slope
    tinv = lambda p, df: abs(t.ppf(p/2, df))
    ts = tinv(0.05, len(x)-2)
    if np.isnan(ts*res.stderr) == False:
        #print(f"slope (95%): {res.slope:.6f} +/- {ts*res.stderr:.6f}")
        fig = plt.figure()
        ax1 = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(2,1,2) #for plotting residuals
        ax1.errorbar(x=x, y=y,yerr=yerr, fmt='.k')
        ax1.plot(x, res.intercept + res.slope*x, label='fitted line')
        ax1.text(0.5, 0.2, f'period: {res.slope:.6f} +/- {ts*res.stderr:.6f}', transform=ax1.transAxes)
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('mid transit time[BTJD]')
        residuals = y - (res.intercept + res.slope*x)
        ax2.errorbar(x=x, y=residuals,yerr=yerr, fmt='.k')
        ax2.plot(x,np.zeros(len(x)), color='red')
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('residuals')
        plt.tight_layout()
        plt.savefig(f'{homedir}/fitting_result/figure/esitimate_period/{KOInumber}.png')
        plt.close()
        return estimated_period
    else:
        estimated_period = period
        return estimated_period

def curve_fitting(each_lc, duration, out, each_lc_list):
    out_transit = each_lc[(each_lc['time'].value < out.params["t0"].value - (duration*0.6)) | (each_lc['time'].value > out.params["t0"].value + (duration*0.6))]
    model = lmfit.models.PolynomialModel()
    poly_params = model.make_params(c0=1, c1=0, c2=0, c3=0, c4=0, c5=0, c6=0, c7=0)
    result = model.fit(out_transit.flux.value, poly_params, x=out_transit.time.value)
    result.plot()
    os.makedirs(f'{homedir}/fitting_result/figure/curvefit/{KOInumber}', exist_ok=True)
    plt.savefig(f'{homedir}/fitting_result/figure/curvefit/{KOInumber}/{KOInumber}_{str(i)}.png')
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

    #normalization
    each_lc.flux = each_lc.flux.value/poly_model(each_lc.time.value)
    each_lc.flux_err = each_lc.flux_err.value/poly_model(each_lc.time.value)
    each_lc_list.append(each_lc)
    return each_lc_list

#if __name__ ==  '__main__':
homedir = '/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani'

df = pd.read_csv(f'{homedir}/KOI1645.csv')
param_df = df
#tpf = lk.search_targetpixelfile('TIC {}'.format(TIC), mission='TESS', cadence="short").download()
while True:
    try:
        search_result = lk.search_lightcurve('KOI-5245', mission='KEPLER', cadence="short")
        #tpf_file = lk.search_targetpixelfile(f'TIC {TIC}', mission='TESS', cadence="short", author='SPOC').download_all(quality_bitmask='default')
        #tpf_file.plot()
        #plt.show()

    except HTTPError:
        print('HTTPError, retry.')
    else:
        break
lc_collection = search_result.download_all()
try:
    lc_collection.plot()
    #plt.savefig(f'{homedir}/lc_collection/TIC{TIC}.png')
    plt.close()
    '''
    for i, lc_now in enumerate(lc_collection):
        mask = lc_now.quality ==0
        lc_now.flux = lc_now.sap_flux
        lc_now = lc_now[mask]
        lc_collection[i] = lc_now
        '''
except AttributeError:
    with open('error_tic.dat', 'a') as f:
        f.write(str(TIC) + '\n')

###恒星変動性の除去
lc = lc_collection.stitch().normalize().remove_nans().remove_outliers() #initialize lc

#x = np.ascontiguousarray(lc.time, dtype=np.float64)
x = lc.time.value
y = np.ascontiguousarray(lc.flux, dtype=np.float64)
yerr = np.ascontiguousarray(lc.flux_err, dtype=np.float64)
mu = np.mean(y)
y = (y / mu - 1) * 1e3
yerr = yerr * 1e3 / mu

results = xo.estimators.lomb_scargle_estimator(
    x, y, max_peaks=1, min_period=5.0, max_period=100.0, samples_per_peak=50
)

peak = results["peaks"][0]
freq, power = results["periodogram"]

with pm.Model() as model:

    # The mean flux of the time series
    mean = pm.Normal("mean", mu=0.0, sd=10.0)

    # A jitter term describing excess white noise
    logs2 = pm.Normal("logs2", mu=2 * np.log(np.mean(yerr)), sd=2.0)

    # A term to describe the non-periodic variability
    logSw4 = pm.Normal("logSw4", mu=np.log(np.var(y)), sd=5.0)
    logw0 = pm.Normal("logw0", mu=np.log(2 * np.pi / 10), sd=5.0)

    # The parameters of the RotationTerm kernel
    logamp = pm.Normal("logamp", mu=np.log(np.var(y)), sd=5.0)
    BoundedNormal = pm.Bound(pm.Normal, lower=0.0, upper=np.log(50))
    logperiod = BoundedNormal("logperiod", mu=np.log(peak["period"]), sd=5.0)
    logQ0 = pm.Normal("logQ0", mu=1.0, sd=10.0)
    logdeltaQ = pm.Normal("logdeltaQ", mu=2.0, sd=10.0)
    mix = xo.distributions.UnitUniform("mix")

    # Track the period as a deterministic
    period = pm.Deterministic("period", tt.exp(logperiod))

    # Set up the Gaussian Process model
    kernel = xo.gp.terms.SHOTerm(log_Sw4=logSw4, log_w0=logw0, Q=1 / np.sqrt(2))
    kernel += xo.gp.terms.RotationTerm(
        log_amp=logamp, period=period, log_Q0=logQ0, log_deltaQ=logdeltaQ, mix=mix
    )
    gp = xo.gp.GP(kernel, x, yerr ** 2 + tt.exp(logs2))

    # Compute the Gaussian Process likelihood and add it into the
    # the PyMC3 model as a "potential"
    pm.Potential("loglike", gp.log_likelihood(y - mean))

    # Compute the mean model prediction for plotting purposes
    pm.Deterministic("pred", gp.predict())

    # Optimize to find the maximum a posteriori parameters
    map_soln = xo.optimize(start=model.test_point)

plt.plot(x, y, "k", label="data")
plt.plot(x, map_soln["pred"], color="C1", label="model")
plt.xlim(x.min(), x.max())
plt.legend(fontsize=10)
plt.xlabel("time [days]")
plt.ylabel("relative flux [ppt]")
plt.show()

#各惑星系の惑星ごとに処理
for index, item in param_df.iterrows():

    '''
    ax = lc.errorbar(label="Kepler-1647")                   # plot() returns a matplotlib axes ...
    trend.plot(ax=ax, color='red', lw=2, label='Trend');  # which we can pass to the next plot() to use the same axes
    plt.show()
    flat.errorbar(label="Kepler-1647");
    plt.show()
    import pdb; pdb.set_trace()
    '''
    duration = item['koi_duration'] / 24
    period = item['koi_period']
    transit_time = item['koi_time0bk']
    transit_start = transit_time - (duration/2)
    transit_end = transit_time + (duration/2)
    KOInumber = 'KOI5245'
    rp = item['koi_prad'] * 0.00916794 #translate to Rsun
    rs = item['koi_srad']
    rp_rs = (rp**2)/(rs**2)
    '''
    bls_period = np.linspace(period*0.6, period*1.5, 10000)
    bls = lc.to_periodogram(method='bls',period=bls_period)#oversample_factor=1)\
    print('planet_b_period = ', bls.period_at_max_power)
    print(f'period = {period}')
    print('planet_b_t0 = ', bls.transit_time_at_max_power)
    print(f'period = {transit_time}')
    print('planet_b_dur = ', bls.duration_at_max_power)
    print(f'period = {duration}')
    duration = bls.duration_at_max_power.value
    period = bls.period_at_max_power.value
    transit_time = bls.transit_time_at_max_power.value
    '''

    """もしもどれかのパラメータがnanだったらそのTIC or TOIを記録して、処理はスキップする。"""
    pdf = pd.Series([duration, period, transit_time], index=['duration', 'period', 'transit_time'])
    if np.sum(pdf.isnull()) != 0:
        with open('error_tic.dat', 'a') as f:
            f.write(f'nan {pdf[pdf.isnull()].index.tolist()}!:{str(TIC)}+ "\n"')
        continue
    print('analysing: ', KOInumber)
    print('judging whether or not transit is included in the data...')
    time.sleep(1)

    """ターゲットの惑星の信号がデータに影響を与えているか判断"""
    _, contain_transit, transit_time_list = clip_transit_hoge(lc, duration, period, transit_time, clip_transit=False)

    """ターゲットの惑星の信号がデータに影響を与えていないなら処理を中断する"""
    if contain_transit == 1:

        ax = lc.scatter()
        #for transit in transit_time_list:

            #ax = lc.scatter()
            #plt.axvline(x=transit, ymax=np.max(lc.flux.value), ymin=np.min(lc.flux.value), color='red')
            #ax.axvline(x=transit, color='blue',alpha=0.7)
            #ax.axvspan(transit-(duration/2), transit+(duration/2), color = "gray", alpha=0.3, hatch="////")
            #ax.set_xlim(transit-duration, transit+duration)
            #plt.show()

        #import pdb; pdb.set_trace()
        #plt.savefig(f'{homedir}/check_transit_timing/TIC{TIC}.png')
        #plt.show()
        plt.close()


        pass

    else:
        print('no transit in data: ', KOInumber)
        continue

    #他の惑星がある場合、データに影響を与えているか判断。ならその信号を除去する。
    if len(param_df.index) != 1:
        print('removing others planet transit in data...')
        time.sleep(1)
        others_duration = param_df[param_df.index!=index]['Planet Transit Duration Value [hours]'].values / 24
        others_period = param_df[param_df.index!=index]['Planet Orbital Period Value [days]'].values
        others_transit_time = param_df[param_df.index!=index]['Planet Transit Midpoint Value [BJD]'].values - 2457000.0 #translate BTJD

        for other_duration, other_period, other_transit_time in zip(others_duration, others_period, others_transit_time):
            if np.any(np.isnan([other_period, other_duration, other_transit_time])):
                continue
            else:
                lc, _, _ = clip_transit_hoge(lc, other_duration, other_period, other_transit_time, clip_transit=True)

    #トランジットがデータに何個あるか判断しその周りのライトカーブデータを作成、カーブフィッティングでノーマライズ
    #fitting using the values of catalog
    folded_lc = lc.fold(period=period , epoch_time=transit_time)
    not_nan_index = np.where(~np.isnan(folded_lc.flux.value))[0].tolist()
    folded_lc = folded_lc[not_nan_index]
    folded_lc, epoch_all_list = detect_transit_epoch(folded_lc, transit_time, period)
    params = transit_params_setting(rp_rs, period)

    #値を格納するリストを定義
    t0dict = {}
    #time_now_arr = []
    outliers = []
    each_lc_list = []

    '''
    """estimate period(2021/12/08現在解析にはestimate periodを使用していない。残余をみる目的)"""
    print('estimate period...')
    time.sleep(1)
    for i, mid_transit_time in enumerate(transit_time_list):
        print(f'epoch: {i}')
        epoch_start = mid_transit_time - (duration*2.5)
        epoch_end = mid_transit_time + (duration*2.5)
        tmp = folded_lc[folded_lc.time_original.value > epoch_start]
        each_lc = tmp[tmp.time_original.value < epoch_end]

        #解析中断条件を満たさないかチェック。トランジットがライトカーブに収まっていて、トランジット中にデータの欠損がない場合のみ解析する
        if len(each_lc) == 0:
            print('> no data in this epoch')
            continue
        abort_list = np.array([transit_case_is4(each_lc, duration, period), aroud_midtransitdata_isexist(each_lc), nospace_in_transit(each_lc, transit_start, transit_end)])
        if np.all(abort_list) == True:
            pass
        else:
            print('> Satisfies the analysis interruption condition')
            continue
        _, _, _,t0dict, _ = transit_fit_and_remove_outliers(each_lc, t0dict, outliers, estimate_period=True, lc_type='each')
    #estimated_periodで再refolding
    if len(t0dict) != 1:
        _ = estimate_period(t0dict, period)
    else:
        estimated_period = period
    #folded_lc = lc.fold(period=estimated_period , epoch_time=transit_time)
    #not_nan_index = np.where(~np.isnan(folded_lc.flux.value))[0].tolist()
    #folded_lc = folded_lc[not_nan_index]
    #folded_lc, epoch_all_list = detect_transit_epoch(folded_lc, transit_time, estimated_period)
    #params = transit_params_setting(rp_rs, estimated_period)
    '''

    """各エポックで外れ値除去、カーブフィッティング"""
    #値を格納するリストの定義
    outliers = []
    each_lc_list = []
    t0list =[]
    print('preprocessing...')
    time.sleep(1)
    #ax = lc.scatter()

    for i, mid_transit_time in enumerate(transit_time_list):
        print(f'epoch: {i}')
        epoch_start = mid_transit_time - (duration*2.5)
        epoch_end = mid_transit_time + (duration*2.5)
        tmp = folded_lc[folded_lc.time_original.value > epoch_start]
        each_lc = tmp[tmp.time_original.value < epoch_end]

        #解析中断条件を満たさないかチェック
        if len(each_lc) == 0:
            print('no data in this epoch')
            continue
        abort_list = np.array([transit_case_is4(each_lc, duration, period), aroud_midtransitdata_isexist(each_lc), nospace_in_transit(each_lc, transit_start, transit_end)])
        if np.all(abort_list) == True:
            pass
        else:
            print('Satisfies the analysis interruption condition')
            continue
        each_lc, _, out, _, no_use_lc = transit_fit_and_remove_outliers(each_lc, t0dict, outliers, estimate_period=False, lc_type='each')
        if no_use_lc == True:
            continue
        else:
            each_lc_list = curve_fitting(each_lc, duration, out, each_lc_list)
    '''
    ax = lc.scatter()
    for t0 in t0list:
        ax.axvline(x=t0, color='red', alpha=0.7, label='aizawa')
    for transit in transit_time_list:
        ax.axvline(x=transit, color='blue',alpha=0.7, label='umetani')
    plt.show()
    import pdb; pdb.set_trace()
    '''

    """folded_lcに対してtransitfit & remove outliers. folded_lcを描画する"""
    print('refolding...')
    time.sleep(1)
    #outliers = []
    try:
        cleaned_lc = vstack(each_lc_list)
    except ValueError:
        pass
    cleaned_lc.sort('time')
    try:
        cleaned_lc, outliers_fold, out, _, _ = transit_fit_and_remove_outliers(cleaned_lc, t0dict, outliers, estimate_period=False)
        flux_model = no_ring_model_transitfit_from_lmparams(out.params, cleaned_lc.time.value, names)
    except ValueError:
        print('no transit!')
        with open('error_tic.dat', 'a') as f:
            f.write('no transit!: ' + 'str(TIC)' + '\n')
        continue
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1) #for plotting transit model and data
    ax2 = fig.add_subplot(2,1,2) #for plotting residuals
    cleaned_lc.errorbar(ax=ax1, color='black', marker='.', zorder=1, label='data')
    ax1.plot(cleaned_lc.time.value, flux_model, label='fitting model', color='red', zorder=2)
    '''
    try:
        outliers_fold = vstack(outliers_fold)
        outliers_fold.errorbar(ax=ax1, label='outliers(folded_lc)', color='blue', marker='.')
    except AttributeError:
        pass
    except ValueError:
        print('no outliers in folded_lc')
        pass
    '''
    ax1.legend()
    ax1.set_title(KOInumber)
    residuals = cleaned_lc - flux_model
    residuals.errorbar(ax=ax2, color='black', ecolor='gray', alpha=0.3,  marker='.', zorder=1)
    ax2.plot(cleaned_lc.time.value, np.zeros(len(cleaned_lc.time)), color='red', zorder=2)
    ax2.set_ylabel('residuals')
    plt.tight_layout()
    #plt.show()
    #import pdb; pdb.set_trace()
    plt.savefig(f'/Users/u_tsubasa/Dropbox/ring_planet_research/folded_lc/figure/{KOInumber}.png')
    plt.close()
    cleaned_lc.write(f'/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/folded_lc_data/{KOInumber}.csv')

    binned_lc = cleaned_lc.bin(time_bin_size=3*u.minute)
    ax = plt.subplot(1,1,1)
    binned_lc.errorbar(ax=ax, color='black')
    ax.set_title(KOInumber)
    plt.savefig(f'/Users/u_tsubasa/Dropbox/ring_planet_research/binned_lc/figure/{KOInumber}.png')
    #plt.show()
    plt.close()
    binned_lc.write(f'/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/binned_lc_data/{KOInumber}.csv')

    print(f'Analysis completed: {KOInumber}')
