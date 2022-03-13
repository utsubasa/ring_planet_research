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

def clip_transit(lc, duration, epoch_list):
    for transit_time in epoch_list:
        transit_start = transit_time - (duration/2)
        transit_end = transit_time + (duration/2)
        case = judge_transit_contain(lc, transit_start, transit_end)
        #print('case:', case)
        #他の惑星処理に使う場合は以下の処理を行う。
        if len(lc[(lc['time'].value > transit_start) & (lc['time'].value < transit_end)]) != 0:
            lc = remove_transit_signal(case, lc, transit_start, transit_end)
        else:
            pass
            #print("don't need clip because no data around transit")

    return lc, transit_time_list

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
        values = [0, period, np.random.uniform(0.01,0.1), np.random.uniform(1.01,20.0), 80.0, 0.5, 90.0, np.random.uniform(0.01,1.0), np.random.uniform(0.01,1.0)]
    else:
        values = [0, period, rp/rs, np.random.uniform(1.01,20.0), 80.0, 0.5, 90.0, np.random.uniform(0.01,1.0), np.random.uniform(0.01,1.0)]
    mins = [-0.7, period*0.9, 0.001, 0.1, 70, 0, 90, 0.0, 0.0]
    maxes = [0.7, period*1.1, 1.0, 100, 110, 1.0, 90, 1.0, 1.0]
    vary_flags = [True, True, True, True, True, True, False, True, True]
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

def transit_fit_and_remove_outliers(lc, t0dict, t0list, outliers, estimate_period=False, lc_type=None):
    #不具合の出るlcはcurvefittingでも弾けるようにno_use_lcを定義
    no_use_lc = False
    while True:
        """transit fitting"""
        flag_time = np.abs(lc.time.value)<1.0
        lc = lc[flag_time]
        t = lc.time.value
        flux = lc.flux.value
        flux_err = lc.flux_err.value
        best_res_dict = {}
        for n in range(10):
            params = transit_params_setting(rp_rs, period)
            out = lmfit.minimize(no_ring_residual_transitfit, params, args=(t, flux, flux_err, names), max_nfev=1000)
            #time.sleep(2)
            best_res_dict[out.chisqr] = out
        out = sorted(best_res_dict.items())[0][1]
        ###lc.time = lc.time - out.params['t0'].value t0を補正する場合に使う
        #print(lmfit.fit_report(out))

        """remove outliers"""
        if lc_type == 'each':
            #print(out.params.pretty_print())
            #time_now_arr.append(0.5 * np.min(each_lc.time_original.value) + 0.5* np.max(each_lc.time_original.value))
            flux_model = no_ring_model_transitfit_from_lmparams(out.params, t, names)
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
                    ax1.plot(t,flux_model, label='fit_model', color='red')
                    try:
                        outliers = vstack(outliers)
                        outliers.errorbar(ax=ax1, color='cyan', label='outliers(each_lc)', marker='.')
                    except ValueError:
                        pass
                    ax1.legend()
                    ax1.set_title(f'chi square/dof: {int(out.chisqr)}/{out.nfree} ')
                    residuals = lc - flux_model
                    residuals.errorbar(ax=ax2, color='black', marker='.')
                    ax2.plot(t,np.zeros(len(t)), label='fitting model', color='red')
                    ax2.set_ylabel('residuals')
                    os.makedirs(f'{homedir}/fitting_result/figure/each_lc/{TOInumber}', exist_ok=True)
                    plt.tight_layout()
                    plt.savefig(f'{homedir}/fitting_result/figure/each_lc/{TOInumber}/{TOInumber}_{str(i)}.png', header=False, index=False)
                    #ax.set_xlim(-1, 1)
                    #plt.show()
                    plt.close()
                else:
                    #pass
                    ###epoch ベースの場合
                    #t0list.append(transit_time+(period*i)+out.params["t0"].value)
                    #t0dict[i] = [transit_time+(period*i)+out.params["t0"].value, out.params["t0"].stderr]
                    ###
                    t0list.append(mid_transit_time+out.params["t0"].value)
                    t0dict[i] = [mid_transit_time+out.params["t0"].value, out.params["t0"].stderr]
                    #t0dict[i] = [out.params["t0"].value, out.params["t0"].stderr]
                    #each_lc = clip_lc
                break
            else:
                #print('removed bins:', len(each_lc[mask]))
                outliers.append(lc[mask])
                lc = lc[~mask]
        else:
            break
    return lc, outliers, out, t0dict, t0list, no_use_lc

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
        import pdb; pdb.set_trace()
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
    poly_params = model.make_params(c0=1, c1=0, c2=0, c3=0, c4=0, c5=0, c6=0, c7=0)
    #poly_params = model.make_params(c0=1, c1=0, c2=0, c3=0)
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
    each_lc.errorbar()
    os.makedirs(f'{homedir}/fitting_result/figure/each_lc/after_curvefit/{TOInumber}', exist_ok=True)
    plt.savefig(f'{homedir}/fitting_result/figure/each_lc/after_curvefit/{TOInumber}/{TOInumber}_{str(i)}.png')
    plt.close()
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
each_lc_anomalylist = [102.01,106.01,114.01,123.01,135.01,150.01,163.01,173.01,349.01,471.01,
                        505.01,625.01,626.01,677.01,738.01,834.01,842.01,845.01,858.01,
                        934.01,987.01,1019.01,1161.01,1163.01,1176.01,1259.01,1264.01,
                        1274.01,1341.01,1845.01,1861.01,1924.01,1970.01,2000.01,2014.01,2021.01,2131.01,
                        2200.01,2222.01,3846.01,4087.01,4486.01]#トランジットが途中で切れてcurvefitによって変なかたちになっているeach_lcを削除したデータ
mtt_shiftlist = [1092.01,129.01,199.01,236.01,758.01,774.01,780.01,822.01,834.01,1050.01,1151.01,1165.01,
                1236.01,1265.01,1270.01,1292.01,1341.01,1721.01,1963.01,2131.01] #mid_transit_time shift　
mtt_shiftlist = [129.01,199.01,236.01,758.01,774.01,780.01,822.01,834.01,1050.01,1151.01,1165.01,1236.01,1265.01,
                1270.01,1292.01,1341.01,1721.01,1963.01,2131.01] #mid_transit_time shift　
no_data_list = [4726.01,372.01,352.01,2617.01,2766.01,2969.01,2989.01,2619.01,2626.01,2624.01,2625.01,
                2622.01,3041.01,2889.01,4543.01,3010.01,2612.01,4463.01,4398.01,4283.01,4145.01,3883.01,
                4153.01,3910.01,3604.01,3972.01,3589.01,3735.01,4079.01,4137.01,3109.01,3321.01,3136.01,
                3329.01,2826.01,2840.01,3241.01,2724.01,2704.01,2803.01,2799.01,2690.01,2745.01,2645.01,
                2616.01,2591.01,2580.01,2346.01,2236.01,2047.01,2109.01,2031.01,2040.01,2046.01,1905.01,
                2055.01,1518.01,1567.01,1355.01,1498.01,1373.01,628.01,1482.01,1580.01,1388.01,1310.01,
                1521.01,1123.01] #short がないか、SPOCがないか

done_list = [4470.01,116.01,105.01,112.01,116.01,190.01,195.01,241.01,398.01,
                413.01,423.01,671.01,744.01,1069.01,1076.01,418.01,391.01,231.01,
                192.01,143.01,766.01,246.01,250.01,567.01,404.01,490.01,1612.01,
                1268.01,1173.01,1670.01,1135.01,1257.01,1181.01,4449.01,1823.01,
                1312.01,1148.01,1165.01,2579.01,1627.01,683.01,1937.01,157.01,
                4606.01,4612.01,4601.01,2218.01,899.01,4623.01,781.01,1682.01,4516.01,
                4535.01,4545.01,1494.01,267.01,2403.01,573.01,272.01,675.01,656.01,232.01,
                481.01,655.01,107.01,483.01,479.01,194.01,477.01,101.01,4486.01,640.01,
                1107.01,1861.01,924.01,159.01,173.01,1951.01,4420.01,3460.01,165.01,201.01,
                1924.01,1059.01,905.01,1934.01,4381.01,1976.01,527.01,1025.01,3492.01,
                212.01,622.01,507.01,780.01,778.01,769.01,4162.01,4059.01,4138.01,3846.01,
                3960.01,3612.01,4140.01,3849.01,3501.01,495.01,501.01,1478.01,615.01,1909.01,
                1012.01,966.01,508.01,811.01,2464.01,511.01,624.01,1936.01,559.01,1130.01,
                264.01,185.01,857.01,182.01,121.01,1425.01,1830.01,665.01,1573.01,1651.01,2126.01,
                1864.01,1295.01,1251.01,1420.01,2127.01,1825.01,2129.01,2197.01,2020.01,2119.01,
                1810.01,1796.01,1811.01,1771.01,1779.01,1845.01,2024.01,1826.01,472.01,1455.01,
                1465.01,1431.01,1300.01,224.01,1150.01,1766.01,585.01,1456.01,453.01,2154.01,
                2017.01,368.01,1385.01,1302.01,1725.01,621.01,1271.01,587.01,1283.01,147.01,
                1299.01,1198.01,1198.01,1815.01,767.01,1676.01,959.01,1714.01,1297.01,1767.01,
                1141.01,1337.01,2025.01,1092.01,1454.01,964.01,590.01,1874.01,1647.01,1419.01,
                1104.01,1248.01,828.01,645.01,1833.01,1721.01,1458.01,820.01,1615.01,2140.01,
                1186.01,818.01,984.01,1182.01,433.01,]

'''
df = df.set_index(['TOI'])
df = df.drop(index=each_lc_anomalylist)
df = df.drop(index=mtt_shiftlist, errors='ignore')
df = df.drop(index=done_list, errors='ignore')
df = df.drop(index=no_data_list, errors='ignore')
df = df.reset_index()
'''
df = df.sort_values('Planet SNR', ascending=False)
df['TOI'] = df['TOI'].astype(str)
TOIlist = df['TOI']
TOIlist = ['2131.01']
#TOIlist = mtt_shiftlist
for TOI in TOIlist:
    TOI = str(TOI)
    param_df = df[df['TOI'] == TOI]
    duration = param_df['Duration (hours)'].values[0] / 24
    period = param_df['Period (days)'].values[0]
    transit_time = param_df['Transit Epoch (BJD)'].values[0] - 2457000.0 #translate BTJD
    transit_start = transit_time - (duration/2)
    transit_end = transit_time + (duration/2)
    TOInumber = 'TOI' + str(param_df['TOI'].values[0])
    rp = param_df['Planet Radius (R_Earth)'].values[0] * 0.00916794 #translate to Rsun
    rs = param_df['Stellar Radius (R_Sun)'].values[0]
    rp_rs = rp/rs
    #tpf = lk.search_targetpixelfile('TIC {}'.format(TIC), mission='TESS', cadence="short").download()
    while True:
        try:
            search_result = lk.search_lightcurve(f'TOI{TOI}', mission='TESS', cadence="short", author='SPOC')
            #tpf_file = lk.search_targetpixelfile(f'TIC {TIC}', mission='TESS', cadence="short", author='SPOC').download_all(quality_bitmask='default')
            #tpf_file.plot()
            #plt.show()

        except HTTPError:
            print('HTTPError, retry.')
        else:
            break
    lc_collection = search_result.download_all()
    lc = lc_collection.stitch() #initialize lc
    '''
    """bls analysis"""
    bls_period = np.linspace(period*0.6, period*1.5, 10000)
    bls = lc.to_periodogram(method='bls',period=bls_period)#oversample_factor=1)\
    print('bls period = ', bls.period_at_max_power)
    print(f'period = {period}')
    print('bls transit time = ', bls.transit_time_at_max_power)
    print(f'transit time = {transit_time}')
    print('bls duration = ', bls.duration_at_max_power)
    print(f'duration = {duration}')
    #duration = bls.duration_at_max_power.value
    #period = bls.period_at_max_power.value
    bls_transit_time = bls.transit_time_at_max_power.value
    catalog_lc = lc.fold(period=period, epoch_time=transit_time)
    bls_lc = lc.fold(period=period, epoch_time=bls_transit_time)
    ax = catalog_lc[np.abs(catalog_lc.time.value)<1.0].scatter(label='catalog t0')
    bls_lc[np.abs(bls_lc.time.value)<1.0].scatter(ax=ax,color='red', label='BLS t0')
    plt.savefig(f'/Users/u_tsubasa/Dropbox/ring_planet_research/comp_bls/{TOInumber}.png')
    plt.close()
    continue
    transit_time = bls_transit_time
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

    """他の惑星がある場合、データに影響を与えているか判断。ならその信号を除去する。"""
    other_p_df = oridf[oridf['TIC ID'] == param_df['TIC ID'].values[0]]

    if len(other_p_df.index) != 1:
        print('removing others planet transit in data...')
        time.sleep(1)
        cliped_lc = lc
        for index, item in other_p_df[other_p_df['TOI']!=float(TOI)].iterrows():
            others_duration = item['Duration (hours)'] / 24
            others_period = item['Period (days)']
            others_transit_time = item['Transit Epoch (BJD)'] - 2457000.0 #translate BTJD
            others_transit_time_list = np.append(np.arange(others_transit_time, lc.time[-1].value, others_period), np.arange(others_transit_time, lc.time[0].value, -others_period))
            others_transit_time_list.sort()
            if np.any(np.isnan([others_period, others_duration, others_transit_time])):
                with open('error_tic.dat', 'a') as f:
                    f.write(f'nan {pdf[pdf.isnull()].index.tolist()}!:{str(TOI)}+ "\n"')
                continue
            else:
                cliped_lc, _ = clip_transit(cliped_lc, others_duration, others_transit_time_list)
        ax = lc.scatter(color='red', label='Other transit signals' )
        cliped_lc.scatter(ax=ax, color='black')
        plt.savefig(f'//Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/other_transit_signals/{TOInumber}.png')
        plt.close()


    #トランジットがデータに何個あるか判断しその周りのライトカーブデータを作成、カーブフィッティングでノーマライズ
    #fitting using the values of catalog
    folded_lc = lc.fold(period=period, epoch_time=transit_time)
    folded_lc = folded_lc.remove_nans()
    folded_lc, epoch_all_list = detect_transit_epoch(folded_lc, transit_time, period)

    """ターゲットの惑星のtransit time listを作成"""
    transit_time_list = np.append(np.arange(transit_time, lc.time[-1].value, period), np.arange(transit_time, lc.time[0].value, -period))
    transit_time_list.sort()

    """各エポックで外れ値除去、カーブフィッティング"""
    #値を格納するリストの定義
    outliers = []
    t0dict = {}
    each_lc_list = []
    t0list =[]
    '''
    ax = lc.scatter()
    for i, mid_transit_time in enumerate(transit_time_list):
        print(f'epoch: {i}')
        epoch_start = mid_transit_time - (duration*2.5)
        epoch_end = mid_transit_time + (duration*2.5)
        tmp = lc[lc.time.value > epoch_start]
        each_lc = tmp[tmp.time.value < epoch_end]
        #ax = lc.scatter()
        #if i == 9:
            #ax.axvline(mid_transit_time)
    plt.show()
    import pdb; pdb.set_trace()
    '''
    '''
    print('fixing t0...')
    time.sleep(1)
    for i, mid_transit_time in enumerate(transit_time_list):
        print(f'epoch: {i}')
        epoch_start = mid_transit_time - (duration*2.5)
        epoch_end = mid_transit_time + (duration*2.5)
        tmp = lc[lc.time.value > epoch_start]
        each_lc = tmp[tmp.time.value < epoch_end]
        #ax = lc.scatter()
        #ax.axvline(mid_transit_time)
        #plt.show()
        each_lc = each_lc.fold(period=period, epoch_time=mid_transit_time).remove_nans()
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
        _, _, _, t0dict, t0list, _ = transit_fit_and_remove_outliers(each_lc, t0dict, t0list, outliers, estimate_period=True, lc_type='each')
    transit_time_list = t0list
    _ = estimate_period(t0dict, period) #TTVを調べる。

    ax = lc.scatter()
    for i, mid_transit_time in enumerate(transit_time_list):
        ax.axvline(mid_transit_time, alpha=0.3)
    plt.savefig(f'/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/check_transit_timing/{TOI}.png')
    plt.close()
    '''

    print('preprocessing...')
    time.sleep(1)

    for i, mid_transit_time in enumerate(transit_time_list):
        print(f'epoch: {i}')
        '''
        if i == 7 or i == 206:
            continue
            '''


        epoch_start = mid_transit_time - (duration*2.5)
        epoch_end = mid_transit_time + (duration*2.5)
        tmp = lc[lc.time.value > epoch_start]
        each_lc = tmp[tmp.time.value < epoch_end]
        each_lc = each_lc.fold(period=period, epoch_time=mid_transit_time).remove_nans()

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
        each_lc, _, out, _, _, no_use_lc = transit_fit_and_remove_outliers(each_lc, t0dict, t0list, outliers, estimate_period=False, lc_type='each')
        if no_use_lc == True:
            continue
        else:
            each_lc_list = curve_fitting(each_lc, duration, out, each_lc_list)
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
        cleaned_lc, outliers_fold, out, _, _, _ = transit_fit_and_remove_outliers(cleaned_lc, t0dict, t0list, outliers, estimate_period=False)
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
    plt.savefig(f'/Users/u_tsubasa/Dropbox/ring_planet_research/folded_lc/figure/{TOInumber}.png')
    #plt.show()
    plt.close()
    cleaned_lc.write(f'/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/folded_lc_data/bls/{TOInumber}.csv')

    binned_lc = cleaned_lc.bin(time_bin_size=3*u.minute)
    ax = plt.subplot(1,1,1)
    binned_lc.errorbar(ax=ax, color='black')
    ax.set_title(TOInumber)
    plt.savefig(f'/Users/u_tsubasa/Dropbox/ring_planet_research/binned_lc/figure/{TOInumber}.png')
    #plt.show()
    plt.close()
    #binned_lc.write(f'/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/binned_lc_data/{TOInumber}.csv')
    print(f'Analysis completed: {TOInumber}')
    with open('./done.csv','a') as f:
        f.write(f'{TOI},')
