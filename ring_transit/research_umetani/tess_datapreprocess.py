# -*- coding: utf-8 -*-
import pdb
from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import lmfit
import lightkurve as lk
from fit_model import model
import warnings
import c_compile_ring
import batman
#import datetime
from decimal import *
import time
#from lightkurve import search_targetpixelfile
from scipy.stats import t, linregress
from astropy.table import Table, vstack
#import astropy.units as u
from decimal import Decimal, ROUND_HALF_UP
import os
#import celerite
#from celerite import terms
#from scipy.optimize import minimize
from astropy.io import ascii

warnings.filterwarnings('ignore')


def bls_analysis(lc, period, transit_time, duration):
    print('bls analysis...')
    time.sleep(1)
    bls_period = np.linspace(10, 50, 10000)
    bls = lc.to_periodogram(method='bls',period=bls_period)#oversample_factor=1)\
    print('bls period = ', bls.period_at_max_power)
    print(f'period = {period}')
    print('bls transit time = ', bls.transit_time_at_max_power)
    print(f'transit time = {transit_time}')
    print('bls duration = ', bls.duration_at_max_power)
    print(f'duration = {duration}')
    bls_transit_time = bls.transit_time_at_max_power.value
    catalog_lc = lc.fold(period=period, epoch_time=transit_time)
    bls_lc = lc.fold(period=period, epoch_time=bls_transit_time)
    ax = catalog_lc[np.abs(catalog_lc.time.value)<1.0].scatter(label='catalog t0')
    bls_lc[np.abs(bls_lc.time.value)<1.0].scatter(ax=ax,color='red', label='BLS t0')
    plt.savefig(f'/Users/u_tsubasa/Dropbox/ring_planet_research/comp_bls/{TOInumber}.png')
    plt.close()
    transit_time = bls_transit_time
    #duration = bls.duration_at_max_power.value
    period = bls.period_at_max_power.value
    
    return transit_time, period

def calc_data_survival_rate(lc, duration):
    data_n = len(lc.flux)
    max_data_n = duration*5*60*24/2 #mid_transit_timeからdurationの前後×2.5 [min]/ 2 min cadence
    data_survival_rate = data_n / max_data_n
    print(f'{data_survival_rate:2f}% data usable')
    #max_data_n = (lc.time_original[-1]-lc.time_original[0])*24*60/2
    return data_survival_rate

def q_to_u_limb(q_arr):
    q1 = q_arr[0]
    q2 = q_arr[1]
    u1 = np.sqrt(q1) * 2 * q2
    u2 = np.sqrt(q1) * (1- 2 * q2)
    return np.array([u1, u2])

def set_params_batman(params_lm, p_names, limb_type ="quadratic"):
    params = batman.TransitParams()       #object to store transit parameters
    params.limb_dark =  limb_type        #limb darkening model
    q_arr = np.zeros(2)
    for i in range(len(p_names)):
        value = params_lm[p_names[i]]
        name = p_names[i]
        if name=="t0":
            params.t0 = value
        if name=="per":
            params.per = value
        if name=="rp":
            params.rp = value
        if name=="a":
            params.a = value
        #if name=="inc":
            #params.inc = value
        if name=="b":
            params.inc =  np.degrees(np.arccos( value / params.a ))
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

def set_params_lm(p_names, values, mins, maxes, vary_flags):
    params = lmfit.Parameters()
    for i in range(len(p_names)):
        if vary_flags[i]:
            params.add(p_names[i], value=values[i], min=mins[i], max = maxes[i], vary = vary_flags[i])
        else:
            params.add(p_names[i], value=values[i], vary = vary_flags[i])
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
def ring_transitfit(params, x, data, eps_data, p_names, return_model=False):
    #start =time.time()
    model = ring_model(x, params.valuesdict())
    chi_square = np.sum(((data-model)/eps_data)**2)
    #print(params)
    #print(chi_square)
    #print(np.max(((data-model)/eps_data)**2))
    if return_model==True:
        return model
    else:
        return (data-model) / eps_data

#リングなしモデルをfitting
def no_ring_transitfit(params, x, data, eps_data, p_names, return_model=False):
    global chi_square
    params_batman = set_params_batman(params, p_names)
    m = batman.TransitModel(params_batman, x)    #initializes model
    model = m.light_curve(params_batman)         #calculates light curve
    chi_square = np.sum(((data-model)/eps_data)**2)
    #print(params)
    #print(chi_square)
    if return_model==True:
        return model
    else:
        return (data-model) / eps_data

def no_ring_transit_and_polynomialfit(params, x, data, eps_data, p_names, return_model=False):
    global chi_square
    params_batman = set_params_batman(params, p_names)
    m = batman.TransitModel(params_batman, x)    #initializes model
    transit_model = m.light_curve(params_batman)         #calculates light curve
    poly_params = params.valuesdict()
    poly_model = np.polynomial.Polynomial([poly_params['c0'],\
                    poly_params['c1'],\
                    poly_params['c2']])
    polynomialmodel = poly_model(x)
    model = transit_model + polynomialmodel -1
    #chi_square = np.sum(((data-model)/eps_data)**2)
    #print(params)
    #print(chi_square)
    if return_model==True:
        return model, transit_model, polynomialmodel
    else:
        return (data-model) / eps_data

def transit_params_setting(rp_rs, period):
    global p_names
    """トランジットフィッティングパラメータの設定"""
    p_names = ["t0", "per", "rp", "a", "b", "ecc", "w", "q1", "q2"]
    if np.isnan(rp_rs):
        values = [np.random.uniform(-0.05,0.05), period, np.random.uniform(0.05,0.1), np.random.uniform(1,10), np.random.uniform(0.0,0.5), 0, 90.0, np.random.uniform(0.1,0.9), np.random.uniform(0.1,0.9)]
    else:
        values = [np.random.uniform(-0.05,0.05), period, rp_rs, np.random.uniform(1,10), np.random.uniform(0.0,0.5), 0, 90.0, np.random.uniform(0.1,0.9), np.random.uniform(0.1,0.9)]
    mins = [-0.2, period*0.8, 0.01, 1, 0, 0, 90, 0.0, 0.0]
    maxes = [0.2, period*1.2, 0.3, 100, 1.0, 0.8, 90, 1.0, 1.0]
    vary_flags = [True, False, True, True, True, False, False, True, True]
    return set_params_lm(p_names, values, mins, maxes, vary_flags)

def calc_obs_transit_time(t0list, t0errlist, num_list, transit_time_list, transit_time_error):
    """return estimated period or cleaned light curve"""
    diff = t0list-transit_time_list
    transit_time_list = transit_time_list[~(diff == 0)]
    t0errlist = t0errlist[~(t0errlist == 0)]
    x = np.array(t0list)[~(diff == 0)]
    y = np.array(x - transit_time_list)*24 #[days] > [hours]
    yerr = np.sqrt( np.square(t0errlist)+np.square(transit_time_error) )*24 #[days] > [hours]
    pd.DataFrame({'x':x, 'O-C':y, 'yerr':yerr}).to_csv(f'{homedir}/fitting_result/data/calc_obs_transit_time/{TOInumber}.csv')
    plt.errorbar(x=x, y=y,yerr=yerr, fmt='.k')
    plt.xlabel('mid transit time[BJD] - 2457000')
    plt.ylabel('O-C(hrs)')
    plt.tight_layout()
    plt.savefig(f'{homedir}/fitting_result/figure/calc_obs_transit_time/{TOInumber}.png')
    plt.close()
    
    x = np.array(num_list)
    y = np.array(t0list)[~(diff == 0)]
    yerr = t0errlist
    try:
        res = linregress(x, y)
    except ValueError:
        print('ValueError: Inputs must not be empty.')
        import pdb; pdb.set_trace()
    estimated_period = res.slope
    tinv = lambda p, df: abs(t.ppf(p/2, df))
    ts = tinv(0.05, len(x)-2)
    
    if np.isnan(ts*res.stderr) == False:
        fig = plt.figure()
        ax1 = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(2,1,2) #for plotting residuals
        try:
            ax1.errorbar(x=x, y=y,yerr=yerr, fmt='.k')
        except TypeError:
            ax1.scatter(x=x, y=y, color='r')
        ax1.plot(x, res.intercept + res.slope*x, label='fitted line')
        ax1.text(0.5, 0.2, f'period: {res.slope:.6f} +/- {ts*res.stderr:.6f}', transform=ax1.transAxes)
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('mid transit time[BJD] - 2457000')
        residuals = y - (res.intercept + res.slope*x)
        try:
            ax2.errorbar(x=x, y=residuals,yerr=yerr, fmt='.k')
        except TypeError:
            ax2.scatter(x=x, y=residuals, color='r')
        ax2.plot(x,np.zeros(len(x)), color='red')
        ax2.set_xlabel('mid transit time[BJD] - 2457000')
        ax2.set_ylabel('residuals')
        plt.tight_layout()
        plt.savefig(f'{homedir}/fitting_result/figure/estimate_period/{TOInumber}.png')
        #plt.savefig(f'{homedir}/fitting_result/figure/calc_/bls/{TOInumber}.png')
        #plt.show()
        plt.close()
        return estimated_period, ts*res.stderr
    else:
        print('np.isnan(ts*res.stderr) == True')
        import pdb;pdb.set_trace()
        estimated_period = period
        return estimated_period
  
def transit_fitting(lc, rp_rs, period, fitting_model=no_ring_transitfit, transitfit_params=None, curvefit_params=None):
    """transit fitting"""
    flag_time = np.abs(lc.time.value)<1.0
    lc = lc[flag_time]
    t = lc.time.value
    flux = lc.flux.value
    flux_err = lc.flux_err.value
    best_res_dict = {} #最も良いreduced chi-squareを出した結果を選別し保存するための辞書
    while len(best_res_dict) == 0:
        for n in range(30):
            params = transit_params_setting(rp_rs, period)
            if transitfit_params != None:
                for p_name in p_names:
                    params[p_name].set(value=transitfit_params[p_name].value)
                    if p_name != 't0':
                        params[p_name].set(vary=False)
            if curvefit_params != None:
                params.add_many(curvefit_params['c0'],
                                curvefit_params['c1'], 
                                curvefit_params['c2'])
            try:
                res = lmfit.minimize(fitting_model, params, args=(t, flux, flux_err, p_names), max_nfev=1000)
                if res.params['t0'].stderr != None:
                    if np.isfinite(res.params['t0'].stderr) and res.redchi < 10:
                    #if res.redchi < 10:
                        red_redchi = abs(res.redchi-1)
                        best_res_dict[red_redchi] = res
            except ValueError:
                print("ValueError")
        if len(best_res_dict) == 0:
            print(TOInumber, i)
            lc.scatter()
            plt.show()
            import pdb;pdb.set_trace()
    res = sorted(best_res_dict.items())[0][1]
    print(f'reduced chisquare: {res.redchi:4f}')
    return res

def clip_outliers(res, lc, outliers, t0list, t0errlist, folded_lc=False, transit_and_poly_fit=False):
    t = lc.time.value
    flux = lc.flux.value
    flux_err = lc.flux_err.value
    if transit_and_poly_fit == True:
        flux_model, transit_model, polynomial_model = no_ring_transit_and_polynomialfit(res.params, t, flux, flux_err, p_names, return_model=True)
    else:
        flux_model = no_ring_transitfit(res.params, t, flux, flux_err, p_names, return_model=True)

    residual_lc = lc.copy()
    residual_lc.flux = np.sqrt(np.square(flux_model - lc.flux))
    _, mask = residual_lc.remove_outliers(sigma=5.0, return_mask=True)
    inverse_mask = np.logical_not(mask)
    print(np.all(inverse_mask))
    if np.all(inverse_mask) == True:
        if folded_lc==True:
            outliers = []
            return lc, outliers, t0list, t0errlist
            
        fig = plt.figure()
        ax1 = fig.add_subplot(2,1,1) #for plotting transit model and data
        ax2 = fig.add_subplot(2,1,2) #for plotting residuals
        lc.errorbar(ax=ax1, color='gray', marker='.', alpha=0.3)
        ax1.plot(t,flux_model, label='fitting model', color='black')
        if transit_and_poly_fit == True:
            ax1.plot(t,transit_model, label='transit model', ls="--", color='blue', alpha=0.5)
            ax1.plot(t,polynomial_model, label='polynomial model', ls="-.", color='red', alpha=0.5)
        try:
            outliers = vstack(outliers)
            outliers.errorbar(ax=ax1, color='cyan', label='outliers(each_lc)', marker='.')
        except ValueError:
            pass
        ax1.legend()
        ax1.set_title(f'chi square/dof: {int(res.chisqr)}/{res.nfree} ')
        residuals = lc - flux_model
        residuals.errorbar(ax=ax2, color='gray', marker='.')
        ax2.plot(t,np.zeros(len(t)), label='fitting model', color='black')
        ax2.set_ylabel('residuals')
        plt.tight_layout()
        if transit_and_poly_fit == False:
            os.makedirs(f'{homedir}/fitting_result/figure/each_lc/transit_fit/{TOInumber}', exist_ok=True)
            plt.savefig(f'{homedir}/fitting_result/figure/each_lc/transit_fit/{TOInumber}/{TOInumber}_{str(i)}.png', header=False, index=False)
            #os.makedirs(f'{homedir}/fitting_result/figure/each_lc/transit_fit/bls/{TOInumber}', exist_ok=True)
            #plt.savefig(f'{homedir}/fitting_result/figure/each_lc/bls/{TOInumber}/{TOInumber}_{str(i)}.png', header=False, index=False)
        else:
            os.makedirs(f'{homedir}/fitting_result/figure/each_lc/transit&poly_fit/{TOInumber}', exist_ok=True)
            plt.savefig(f'{homedir}/fitting_result/figure/each_lc/transit&poly_fit/{TOInumber}/{TOInumber}_{str(i)}.png', header=False, index=False)
            #os.makedirs(f'{homedir}/fitting_result/figure/each_lc/transit&poly_fit/bls/{TOInumber}', exist_ok=True)
            #plt.savefig(f'{homedir}/fitting_result/figure/each_lc/bls/{TOInumber}/{TOInumber}_{str(i)}.png', header=False, index=False)
        plt.close()
        t0list.append(res.params["t0"].value+mid_transit_time)
        t0errlist.append(res.params["t0"].stderr)
        outliers = []
    else:
        #print('removed bins:', len(each_lc[mask]))
        outliers.append(lc[mask])
        lc = lc[~mask]
    plt.close()
    return lc, outliers, t0list, t0errlist

def curve_fitting(each_lc, duration, res=None):
    if res != None:
        out_transit = each_lc[(each_lc['time'].value < res.params["t0"].value - (duration*0.7)) | (each_lc['time'].value > res.params["t0"].value + (duration*0.7))]
    else:
        out_transit = each_lc[(each_lc['time'].value < -(duration*0.7)) | (each_lc['time'].value > (duration*0.7))]
    model = lmfit.models.PolynomialModel(degree=2)
    #poly_params = model.make_params(c0=0, c1=0, c2=0, c3=0, c4=0, c5=0, c6=0, c7=0)
    poly_params = model.make_params(c0=1, c1=0, c2=0)
    result = model.fit(out_transit.flux.value, poly_params, x=out_transit.time.value)
    result.plot()
    os.makedirs(f'{homedir}/fitting_result/figure/curvefit/{TOInumber}', exist_ok=True)
    plt.savefig(f'{homedir}/fitting_result/figure/curvefit/{TOInumber}/{TOInumber}_{str(i)}.png')
    #os.makedirs(f'{homedir}/fitting_result/figure/curvefit/bls/{TOInumber}', exist_ok=True)
    #plt.savefig(f'{homedir}/fitting_result/figure/curvefit/bls/{TOInumber}/{TOInumber}_{str(i)}.png')
    #plt.show()
    plt.close()

    return result

def polynomial_normalize(each_lc, poly_params):
    poly_model = np.polynomial.Polynomial([poly_params['c0'].value,\
                poly_params['c1'].value,\
                poly_params['c2'].value])
    '''
    poly_model = np.polynomial.Polynomial([result.params.valuesdict()['c0'],\
                    result.params.valuesdict()['c1'],\
                    result.params.valuesdict()['c2'],\
                    result.params.valuesdict()['c3'],\
                    result.params.valuesdict()['c4'],\
                    result.params.valuesdict()['c5'],\
                    result.params.valuesdict()['c6'],\
                    result.params.valuesdict()['c7']])
    '''
    #normalization
    each_lc.flux = each_lc.flux.value/poly_model(each_lc.time.value)
    each_lc.flux_err = each_lc.flux_err.value/poly_model(each_lc.time.value)
    each_lc.errorbar()
    os.makedirs(f'{homedir}/fitting_result/figure/each_lc/after_curvefit/{TOInumber}', exist_ok=True)
    plt.savefig(f'{homedir}/fitting_result/figure/each_lc/after_curvefit/{TOInumber}/{TOInumber}_{str(i)}.png')
    #os.makedirs(f'{homedir}/fitting_result/figure/each_lc/after_curvefit/bls/{TOInumber}', exist_ok=True)
    #plt.savefig(f'{homedir}/fitting_result/figure/each_lc/after_curvefit/bls/{TOInumber}/{TOInumber}_{str(i)}.png')
    plt.close()
    os.makedirs(f'{homedir}/fitting_result/data/each_lc/{TOInumber}', exist_ok=True)
    each_lc.write(f'{homedir}/fitting_result/data/each_lc/{TOInumber}/{TOInumber}_{str(i)}.csv')
    return each_lc

def folding_lc_from_csv(filedir, TOInumber):
    outliers = []
    t0list = []
    t0errlist = []
    each_lc_list = []
    try:
        each_lc_list=[]
        total_lc_csv = os.listdir(f'{filedir}/{TOInumber}/')
        total_lc_csv = [i for i in total_lc_csv if 'TOI' in i]
        for each_lc_csv in total_lc_csv:
            each_table = ascii.read(f'{filedir}/{TOInumber}/{each_lc_csv}')
            each_lc = lk.LightCurve(data=each_table)
            each_lc_list.append(each_lc)
    except ValueError:
        pass
    cleaned_lc = vstack(each_lc_list)
    cleaned_lc.sort('time')

    while True:
        res = transit_fitting(cleaned_lc, rp_rs, period, fitting_model=no_ring_transitfit)
        cleaned_lc, outliers, t0list, t0errlist = clip_outliers(res, cleaned_lc, outliers, t0list, t0errlist, folded_lc=True)
        if len(outliers) == 0:
            break 
        
    flux_model = no_ring_transitfit(res.params, cleaned_lc.time.value, cleaned_lc.flux.value, cleaned_lc.flux_err.value, p_names, return_model=True)
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1) #for plotting transit model and data
    ax2 = fig.add_subplot(2,1,2) #for plotting residuals
    cleaned_lc.errorbar(ax=ax1, color='black', marker='.', zorder=1, label='data')
    ax1.plot(cleaned_lc.time.value, flux_model, label='fitting model', color='red', zorder=2)
    ax1.legend()
    ax1.set_title(TOInumber)
    residuals = cleaned_lc - flux_model
    residuals.errorbar(ax=ax2, color='black', ecolor='gray', alpha=0.3,  marker='.', zorder=1)
    ax2.plot(cleaned_lc.time.value, np.zeros(len(cleaned_lc.time)), color='red', zorder=2)
    ax2.set_ylabel('residuals')
    plt.tight_layout()
    plt.savefig(f'/Users/u_tsubasa/Dropbox/ring_planet_research/folded_lc/figure/{TOInumber}.png')
    #plt.savefig(f'/Users/u_tsubasa/Dropbox/ring_planet_research/folded_lc/figure/bls/{TOInumber}.png')
    #plt.show()
    plt.close()
    #cleaned_lc.write(f'/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/folded_lc_data/bls/{TOInumber}.csv')
    cleaned_lc.write(f'/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/fitting_result/data/folded_lc/{TOInumber}.csv')

    return res
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
                1521.01,1123.01, 1519.01, 1427.01, 1371.01, 1365.01, 1397.01] #short がないか、SPOCがないか
no_perioddata_list = [1134.01,1897.01,2423.01,2666.01,4465.01]#exofopの表にperiodの記載無し。1567.01,1656.01もperiodなかったがこちらはcadence=’short’のデータなし。
no_signal_list = [2218.01,212.01,1823.01] #トランジットのsignalが無いか、ノイズに埋もれて見えない
multiplanet_list = [1670.01, 201.01, 822.01]#, 1130.01]
startrend_list = [4381.01, 1135.01, 1025.01, 212.01, 1830.01, 2119.01, 224.01]
flare_list = [212.01, 2119.01, 1779.01]
two_epoch_list = [671.01, 1963.01, 1283.01, 758.01, 1478.01, 3501.01, 964.01, 845.01, 121.01,1104.01, 811.01, 3492.01]

#done_list = [4470.01,495.01,423.01,398.01,165.01,1148.01,157.01,1682.01,1612.01,112.01,656.01]
done_list = os.listdir('/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/fitting_result/data/folded_lc/modelresult/2ndloop/transit&poly')
done_list = [s for s in done_list if 'TOI' in s]
done_list = [s.lstrip('TOI') for s in done_list ]
#done_list = [float(s.strip('.png')) for s in done_list]
done_list = [float(s) for s in done_list]

df = df.set_index(['TOI'])
#df = df.drop(index=each_lc_anomalylist)
#df = df.drop(index=mtt_shiftlist, errors='ignore')
df = df.drop(index=done_list, errors='ignore')
df = df.drop(index=no_data_list, errors='ignore')
df = df.drop(index=multiplanet_list, errors='ignore')
#df = df.drop(index=startrend_list, errors='ignore')
df = df.drop(index=flare_list, errors='ignore')
df = df.drop(index=two_epoch_list, errors='ignore')
#df = df.drop(index=no_signal_list, errors='ignore')
df = df.reset_index()

df = df.sort_values('Planet SNR', ascending=False)
df['TOI'] = df['TOI'].astype(str)
TOIlist = df['TOI']
for TOI in [224.01]:
#for TOI in TOIlist:
    if TOI=='1823.01' or TOI=='1833.01' or TOI=='2218.01':
        continue
    TOI = str(TOI)
    param_df = df[df['TOI'] == TOI]
    duration = param_df['Duration (hours)'].values[0] / 24
    period = param_df['Period (days)'].values[0]
    transit_time = param_df['Transit Epoch (BJD)'].values[0] - 2457000.0 #translate BTJD
    transit_time_error = param_df['Transit Epoch error'].values[0]
    transit_start = transit_time - (duration/2)
    transit_end = transit_time + (duration/2)
    TOInumber = 'TOI' + str(param_df['TOI'].values[0])
    rp = param_df['Planet Radius (R_Earth)'].values[0] * 0.00916794 #translate to Rsun
    rs = param_df['Stellar Radius (R_Sun)'].values[0]
    rp_rs = rp/rs

    """もしもduration, period, transit_timeどれかのパラメータがnanだったらそのTOIを記録して、処理はスキップする"""
    if np.sum(np.isnan([duration, period, transit_time])) != 0:
        #with open('nan3params_toi.dat', 'a') as f:
            #f.write(f'{TOInumber}: {np.isnan([duration, period, transit_time])}\n')
        continue

    print('analysing: ', TOInumber)
    search_result = lk.search_lightcurve(f'TOI{TOI}', mission='TESS', cadence="short", author='SPOC')
    lc_collection = search_result.download_all()
    #lc_collection = search_result.download(1)
    
    lc = lc_collection.stitch().remove_nans() #initialize lc
    #lc = lc_collection.remove_nans().normalize() #initialize lc
    '''
    lc.scatter()
    plt.show()
    import pdb;pdb.set_trace()
    '''
    

    """bls analysis"""
    #transit_time, period = bls_analysis(lc, period, transit_time, duration)

    """他の惑星がある場合、データにそのトランジットが含まれているかを判断し、与えているならその信号を除去する。
    print('judging whether other planet transit is included in the data...')
    time.sleep(1)
    other_p_df = oridf[oridf['TIC ID'] == param_df['TIC ID'].values[0]]
    if len(other_p_df.index) != 1:
        lc = remove_others_transit(lc, oridf, param_df, other_p_df, TOI)
    """

    """ターゲットの惑星のtransit time listを作成"""
    transit_time_list = np.append(np.arange(transit_time, lc.time[-1].value, period), np.arange(transit_time, lc.time[0].value, -period))
    transit_time_list = np.unique(transit_time_list)
    transit_time_list.sort()

    """各エポックで外れ値除去、カーブフィッティング"""
    #値を格納するリストの定義
    outliers = []
    t0list = []
    t0errlist = []
    num_list = []
    #ax = lc.scatter()
    for i, mid_transit_time in enumerate(transit_time_list):
        print(f'preprocessing...epoch: {i}')
        """
        ax.axvline(x=mid_transit_time)
        continue
        """
        #if i == 203:
            #continue  
        epoch_start = mid_transit_time - (duration*2.5)
        epoch_end = mid_transit_time + (duration*2.5)
        tmp = lc[lc.time.value > epoch_start]
        each_lc = tmp[tmp.time.value < epoch_end]
        each_lc = each_lc.fold(period=period, epoch_time=mid_transit_time).normalize().remove_nans()

        """解析中断条件を満たさないかチェック"""
        data_survival_rate = calc_data_survival_rate(each_lc, duration)
        if data_survival_rate < 0.9:
            if data_survival_rate != 0.0:
                ax = each_lc.errorbar()
                ax.set_title(f'{data_survival_rate:4f} useable')
                os.makedirs(f'{homedir}/fitting_result/figure/error_lc/under_90%_data/calc_t0/{TOInumber}', exist_ok=True)
                plt.savefig(f'{homedir}/fitting_result/figure/error_lc/under_90%_data/calc_t0/{TOInumber}/{TOInumber}_{str(i)}.png')
                plt.close()
            t0list.append(mid_transit_time)
            t0errlist.append(np.nan)
            continue
        else:
            num_list.append(i)

        curvefit_res = curve_fitting(each_lc, duration)
        each_lc = polynomial_normalize(each_lc, curvefit_res.params) 
        while True:
            res = transit_fitting(each_lc, rp_rs, period)
            #res = transit_fitting(each_lc, rp_rs, period, fitting_model=no_ring_transit_and_polynomialfit, transitfit_params=res.params, curvefit_params=curvefit_params)
            os.makedirs(f'{homedir}/fitting_result/data/each_lc/modelresult/1stloop/transit/{TOInumber}', exist_ok=True)
            with open(f'{homedir}/fitting_result/data/each_lc/modelresult/1stloop/transit/{TOInumber}/{TOInumber}_{str(i)}.txt', 'a') as f:
                print(lmfit.fit_report(res), file = f)
            each_lc, outliers, t0list, t0errlist = clip_outliers(res, each_lc, outliers, t0list, t0errlist)
            if len(outliers) == 0:
                break 
            else:
                pass
        curvefit_res = curve_fitting(each_lc, duration, res)
        os.makedirs(f'{homedir}/fitting_result/data/each_lc/modelresult/1stloop/curvefit/{TOInumber}', exist_ok=True)
        with open(f'{homedir}/fitting_result/data/each_lc/modelresult/1stloop/curvefit/{TOInumber}/{TOInumber}_{str(i)}.txt', 'a') as f:
            print(lmfit.fit_report(res), file = f)
        _ = polynomial_normalize(each_lc, curvefit_res.params) 
    if len(t0list) <= 2:
        with open('two_period_toi.dat', 'a') as f:
            f.write(f'{TOI}\n')
        continue
    period, period_err = calc_obs_transit_time(t0list, t0errlist, num_list, transit_time_list, transit_time_error) #calc_obs_transit_timeを調べる

    continue
    """folded_lcに対してtransitfit & remove outliers. folded_lcを描画する"""
    print('folding and calculate duration...')
    time.sleep(1)
    fold_res = folding_lc_from_csv(f'{homedir}/fitting_result/data/each_lc/calc_t0', TOInumber)
    a_rs = fold_res.params['a'].value
    b = fold_res.params['b'].value
    inc = np.arccos( b / a_rs )
    if np.isnan(rp_rs):
        rp_rs = fold_res.params['rp'].value
    duration = (period/np.pi)*np.arcsin( (1/a_rs)*( np.sqrt(np.square(1+rp_rs) - np.square(b))/np.sin(inc) ) )
    obs_t0_idx = np.abs(np.asarray(t0list) - transit_time).argmin()
    os.makedirs(f'{homedir}/fitting_result/data/folded_lc/modelresult/1stloop/transit/{TOInumber}', exist_ok=True)
    with open(f'{homedir}/fitting_result/data/folded_lc/modelresult/1stloop/transit/{TOInumber}/{TOInumber}_folded.txt', 'a') as f:
        print(lmfit.fit_report(fold_res), file = f)
        print(f'calculated duration[day]: {duration}', file = f)
        print(f'obs_transit_time[day]: {t0list[obs_t0_idx]}', file = f)
        print(f'obs_transit_time_err[day]: {t0errlist[obs_t0_idx]}', file = f)
        print(f'obs_period[day]: {period}', file = f)
        print(f'obs_period_err[day]: {period_err}', file = f)





















            
    """durationの値、惑星パラメータをfixして、各トランジットエポックでベースライン（多項式フィッティング）とt0を動かしてフィッティングする。"""
    obs_t0_list = t0list
    #値を格納するリストの定義
    outliers = []
    t0list = []
    t0errlist = []
    num_list = []
    
    for i, mid_transit_time in enumerate(obs_t0_list):
        print(f'reprocessing...epoch: {i}')
        #if i == 203:
            #continue  
        epoch_start = mid_transit_time - (duration*2.5)
        epoch_end = mid_transit_time + (duration*2.5)
        tmp = lc[lc.time.value > epoch_start]
        each_lc = tmp[tmp.time.value < epoch_end]
        each_lc = each_lc.fold(period=period, epoch_time=mid_transit_time).normalize().remove_nans()
        
        """解析中断条件を満たさないかチェック"""
        data_survival_rate = calc_data_survival_rate(each_lc, duration)
        if data_survival_rate < 0.9:
            if data_survival_rate != 0.0:
                ax = each_lc.errorbar()
                ax.set_title(f'{data_survival_rate:4f} useable')
                os.makedirs(f'{homedir}/fitting_result/figure/error_lc/under_90%_data/obs_t0/{TOInumber}', exist_ok=True)
                plt.savefig(f'{homedir}/fitting_result/figure/error_lc/under_90%_data/obs_t0/{TOInumber}/{TOInumber}_{str(i)}.png')
                plt.close()
            t0list.append(mid_transit_time)
            t0errlist.append(np.nan)
            continue
        else:
            num_list.append(i)

        curvefit_res = curve_fitting(each_lc, duration)
        while True:
            res = transit_fitting(each_lc, rp_rs, period, fitting_model=no_ring_transit_and_polynomialfit, transitfit_params=fold_res.params, curvefit_params=curvefit_res.params)
            os.makedirs(f'{homedir}/fitting_result/data/each_lc/modelresult/2ndloop/transit&poly/{TOInumber}', exist_ok=True)
            with open(f'{homedir}/fitting_result/data/each_lc/modelresult/2ndloop/transit&poly/{TOInumber}/{TOInumber}_{str(i)}.txt', 'a') as f:
                print(lmfit.fit_report(res), file = f)
            each_lc, outliers, t0list, t0errlist = clip_outliers(res, each_lc, outliers, t0list, t0errlist, transit_and_poly_fit=True)
            if len(outliers) == 0:
                break 
            else:
                pass
        polynomial_normalize(each_lc, res.params)
        #処理したeach_lcは指定されたディレクトリに保存される。

    #calc_obs_transit_time(t0list, t0errlist, num_list, transit_time_list, transit_time_error) #calc_obs_transit_timeを調べる

    """最終的なfolded_lcを生成する。"""
    print('refolding...')
    time.sleep(1)
    fold_res = folding_lc_from_csv(f'{homedir}/fitting_result/data/each_lc', TOInumber)
    os.makedirs(f'{homedir}/fitting_result/data/folded_lc/modelresult/2ndloop/transit&poly/{TOInumber}', exist_ok=True)
    with open(f'{homedir}/fitting_result/data/folded_lc/modelresult/2ndloop/transit&poly/{TOInumber}/{TOInumber}_folded.txt', 'a') as f:
        print(lmfit.fit_report(fold_res), file = f)
    print(f'Analysis completed: {TOInumber}')

#import pdb;pdb.set_trace()