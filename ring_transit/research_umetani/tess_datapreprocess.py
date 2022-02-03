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
                #print("don't need clip because no data around transit")
                pass
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
        values = [0, period, rp/rs, np.random.uniform(1.01,20.0), 80.0, 0.5, 90.0, np.random.uniform(0.01,1.0), np.random.uniform(0.01,1.0)]
    mins = [-0.7, period*0.9, 0.001, 0.1, 70, 0, 90, 0.0, 0.0]
    maxes = [0.7, period*1.1, 1.0, 100, 110, 0, 90, 1.0, 1.0]
    vary_flags = [True, False, True, True, True, False, False, False, False]
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
        plt.savefig(f'{homedir}/fitting_result/figure/error_lc/{TOInumber}_{str(i)}.png', header=False, index=False)
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

            best_res_dict = {}
            for n in range(20):
                params = transit_params_setting(rp_rs, period)
                out = lmfit.minimize(no_ring_residual_transitfit, params, args=(time, flux, flux_err, names), max_nfev=1000)
                best_res_dict[out.chisqr] = out
            out = sorted(best_res_dict.items())[0][1]
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
                            os.makedirs(f'{homedir}/fitting_result/figure/each_lc/{TOInumber}', exist_ok=True)
                            plt.tight_layout()
                            plt.savefig(f'{homedir}/fitting_result/figure/each_lc/{TOInumber}/{TOInumber}_{str(i)}.png', header=False, index=False)
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
                plt.savefig(f'{homedir}/fitting_result/figure/error_lc/{TOInumber}_{str(i)}.png', header=False, index=False)
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
        plt.savefig(f'{homedir}/fitting_result/figure/esitimate_period/{TOInumber}.png')
        plt.close()
        return estimated_period
    else:
        estimated_period = period
        return estimated_period

def curve_fitting(each_lc, duration, out, each_lc_list):
    out_transit = each_lc[(each_lc['time'].value < out.params["t0"].value - (duration*0.7)) | (each_lc['time'].value > out.params["t0"].value + (duration*0.7))]
    model = lmfit.models.PolynomialModel()
    #poly_params = model.make_params(c0=1, c1=0, c2=0, c3=0, c4=0, c5=0, c6=0, c7=0)
    poly_params = model.make_params(c0=1, c1=0, c2=0, c3=0)
    result = model.fit(out_transit.flux.value, poly_params, x=out_transit.time.value)
    result.plot()
    os.makedirs(f'{homedir}/fitting_result/figure/curvefit/{TOInumber}', exist_ok=True)
    plt.savefig(f'{homedir}/fitting_result/figure/curvefit/{TOInumber}/{TOInumber}_{str(i)}.png')
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

def remove_GP(lc): #remove the gaussian process from lc.stitch
    def neg_log_like(params, y, gp):
        gp.set_parameter_vector(params)
        return -gp.log_likelihood(y)

    y = lc.flux.value
    yerr = lc.flux_err.value
    t = lc.time.value

    # A non-periodic component
    Q = 1.0 / np.sqrt(2.0)
    w0 = 3.0
    S0 = np.var(y) / (w0 * Q)
    bounds = dict(log_S0=(-15, 15), log_Q=(-15, 15), log_omega0=(-15, 15))
    kernel = terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0),
                           bounds=bounds)
    kernel.freeze_parameter("log_Q")  # We don't want to fit for "Q" in this term

    # A periodic component
    Q = 1.0
    w0 = 3.0
    S0 = np.var(y) / (w0 * Q)
    kernel += terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0),
                            bounds=bounds)
    gp = celerite.GP(kernel, mean=np.mean(y))
    gp.compute(t, yerr)

    initial_params = gp.get_parameter_vector()
    bounds = gp.get_parameter_bounds()
    r = minimize(neg_log_like, initial_params, method="L-BFGS-B", bounds=bounds, args=(y, gp))
    gp.set_parameter_vector(r.x)
    print(r)
    pred_mean, pred_var = gp.predict(y, t, return_var=True)
    pred_std = np.sqrt(pred_var)
    color = "#ff7f0e"
    plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
    plt.plot(t, pred_mean, color=color)
    plt.fill_between(t, pred_mean+pred_std, pred_mean-pred_std, color=color, alpha=0.3,
                     edgecolor="none")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


#if __name__ ==  '__main__':
homedir = '/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani'

oridf = pd.read_csv(f'{homedir}/exofop_tess_tois.csv')
df = oridf[oridf['Planet SNR']>100]
df['TOI'] = df['TOI'].astype(str)
TOIlist = df['TOI']
TOIlist = ['102.01']
for TOI in TOIlist:
    param_df = df[df['TOI'] == TOI]
    #tpf = lk.search_targetpixelfile('TIC {}'.format(TIC), mission='TESS', cadence="short").download()
    while True:
        try:
            search_result = lk.search_lightcurve(f'TOI {TOI}', mission='TESS', cadence="short", author='SPOC')
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
        #plt.savefig(f'{homedir}/lc_collection/TOI{TOI}.png')
        plt.close()
    except AttributeError:
        with open('error_tic.dat', 'a') as f:
            f.write(TOInumber + '\n')
        continue

    #各惑星系の惑星ごとに処理
    for index, item in param_df.iterrows():
        lc = lc_collection.stitch() #initialize lc
        import pdb; pdb.set_trace()
        duration = item['Duration (hours)'] / 24
        period = item['Period (days)']
        transit_time = item['Transit Epoch (BJD)'] - 2457000.0 #translate BTJD
        transit_start = transit_time - (duration/2)
        transit_end = transit_time + (duration/2)
        TOInumber = 'TOI' + str(item['TOI'])
        rp = item['Planet Radius (R_Earth)'] * 0.00916794 #translate to Rsun
        rs = item['Stellar Radius (R_Sun)']
        rp_rs = rp/rs
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
                f.write(f'nan {pdf[pdf.isnull()].index.tolist()}!:{str(TOI)}+ "\n"')
            continue
        print('analysing: ', TOInumber)
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
            print('no transit in data: ', TOInumber)
            continue

        #他の惑星がある場合、データに影響を与えているか判断。ならその信号を除去する。
        other_p_df = oridf[oridf['TIC ID'] == param_df['TIC ID'].values[0]]
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
        folded_lc = lc.fold(period=period, epoch_time=transit_time)
        folded_lc = folded_lc.remove_nans()
        folded_lc, epoch_all_list = detect_transit_epoch(folded_lc, transit_time, period)


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

        for i, mid_transit_time in enumerate(epoch_all_list):
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
                f.write('no transit!: ' + 'str(TOI)' + '\n')
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
        ax1.set_title(TOInumber)
        residuals = cleaned_lc - flux_model
        residuals.errorbar(ax=ax2, color='black', ecolor='gray', alpha=0.3,  marker='.', zorder=1)
        ax2.plot(cleaned_lc.time.value, np.zeros(len(cleaned_lc.time)), color='red', zorder=2)
        ax2.set_ylabel('residuals')
        plt.tight_layout()
        #plt.show()
        #import pdb; pdb.set_trace()
        plt.savefig(f'/Users/u_tsubasa/Dropbox/ring_planet_research/folded_lc/figure/{TOInumber}.png')
        plt.close()
        cleaned_lc.write(f'/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/folded_lc_data/{TOInumber}.csv')

        binned_lc = cleaned_lc.bin(time_bin_size=3*u.minute)
        ax = plt.subplot(1,1,1)
        binned_lc.errorbar(ax=ax, color='black')
        ax.set_title(TOInumber)
        plt.savefig(f'/Users/u_tsubasa/Dropbox/ring_planet_research/binned_lc/figure/{TOInumber}.png')
        #plt.show()
        plt.close()
        binned_lc.write(f'/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/binned_lc_data/{TOInumber}.csv')

        print(f'Analysis completed: {TOInumber}')
    """
    for i, epoch_now in enumerate(epoch_all_list):
        print(f'epoch: {epoch_now}')
        flag = folded_lc.epoch_all == epoch_now
        each_lc = folded_lc[flag]
        #ax = each_lc.scatter()
        ax = plt.subplot(1,1,1)
        ax.scatter(each_lc.time_original.value, each_lc.flux.value)
        ax.set_title('epoch based')
        #ax.axvspan(each_lc.time_original.value[0], each_lc.time_original.value[-1], color = "gray", alpha=0.3, hatch="////")
        #ax.axvline(x=np.median(each_lc.time_original.value), color='blue',alpha=0.7)
        os.makedirs(f'{homedir}/fitting_result/figure/each_lc/before_preprocess/epoch_based/{TOInumber}', exist_ok=True)
        #plt.savefig(f'{homedir}/fitting_result/figure/each_lc/before_preprocess/epoch_based/{TOInumber}/{TOInumber}_{str(i)}.png', header=False, index=False)
        plt.show()
        plt.close()
        continue

    continue
    """
    '''
    for i, epoch_now in enumerate(epoch_all_list):
        print(f'epoch: {epoch_now}')
        flag = folded_lc.epoch_all == epoch_now
        each_lc = folded_lc[flag]
        if epoch_now == 178 and TOInumber == 'TOI187.01':
            continue
        if i == 11:
            ax = plt.subplot(1,1,1)
            ax.scatter(each_lc.time_original.value, each_lc.flux.value)
            ax.set_title('mid t based')
            #ax.axvline(x=np.median(each_lc.time_original.value), color='blue',alpha=0.7)
            #os.makedirs(f'{homedir}/fitting_result/figure/each_lc/before_preprocess/ori_t_based/{TOInumber}', exist_ok=True)
            #plt.savefig(f'{homedir}/fitting_result/figure/each_lc/before_preprocess/ori_t_based/{TOInumber}/{TOInumber}_{str(i)}.png', header=False, index=False)
            plt.show()
            #plt.close()
            import pdb; pdb.set_trace()
    '''
