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
import astropy.units as u

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
    print(f'transit_time: {transit_time}')
    transit_start = transit_time - (duration/2)
    transit_end = transit_time + (duration/2)

    if transit_time < np.median(lc['time'].value):
        while judge_transit_contain(lc, transit_start, transit_end) < 6:
            print(f'transit_time: {transit_time}')
            transit_start = transit_time - (duration/2)
            transit_end = transit_time + (duration/2)
            case = judge_transit_contain(lc, transit_start, transit_end)
            print('case:', case)
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
                print("don't need clip because no data around transit")
            transit_time = transit_time + period
    elif transit_time > np.median(lc['time'].value):
        while judge_transit_contain(lc, transit_start, transit_end) > 1:
            print(f'transit_time: {transit_time}')
            transit_start = transit_time - (duration/2)
            transit_end = transit_time + (duration/2)
            case = judge_transit_contain(lc, transit_start, transit_end)
            print('case:', case)
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
    if transit_end < lc.time[0].value: # ||-----
        case = 1
    elif transit_start < lc.time[0].value and lc.time[0].value < transit_end and transit_end < lc.time[-1].value: # |--|---
        case = 2
    elif transit_start < lc.time[0].value and lc.time[-1].value < transit_end: # |-----|
        case = 3
    elif lc.time[0].value < transit_start and transit_start < lc.time[-1].value and lc.time[0].value < transit_end and transit_end < lc.time[-1].value: # --|-|--
        case = 4
    elif lc.time[0].value < transit_start and transit_start < lc.time[-1].value and lc.time[-1].value < transit_end: # ---|--|
        case = 5
    elif lc.time[-1].value < transit_start: # -----||
        case = 6
    else:
        print('unexcepted case')
        import pdb; pdb.set_trace()
    return case


def remove_transit_signal(case, lc, transit_start, transit_end):
    if case == 1: # ||-----
        pass
    elif case == 2: # |--|---
        lc = lc[~(lc['time'].value < transit_start)]
    elif case == 3: # |-----|
        #with open('huge_transit.csv') as f:
            #f.write()
        print('huge !')
        #記録する
    elif case == 4: # --|-|--
        #lc = vstack([lc[lc['time'].value < transit_start], lc[lc['time'].value > transit_end]])
        lc = lc[(lc['time'].value < transit_start) | (lc['time'].value > transit_end)]

    elif case == 5: # ---|--|
        lc = lc[lc['time'].value < transit_start]
    elif case == 6: # -----||
        pass

    return lc


def preprocess_each_lc(lc, duration, period, transit_time_list, TOInumber):
    transit_time_list = np.array(transit_time_list)
    n_transit = len(transit_time_list)
    print('n_transit: ', n_transit)
    half_duration = (duration/2)
    twice_duration = (duration*2) #durationを2倍、単位をday→mi
    lc_cut_point = half_duration + twice_duration
    lc_list=[]
    for i, mid_transit in enumerate(transit_time_list):
        print(f'No.{i} transit: ')
        transit_start = mid_transit - lc_cut_point
        transit_end = mid_transit + lc_cut_point
        """
        each_lc_pre = lc[:start]
        each_lc_mid = lc[start:end]
        each_lc_post = lc[end:]
        print('before clip length: ', len(each_lc.flux))
        for each_lc in [each_lc_pre, each_lc_post]:
            clip_lc = each_lc.normalize().copy()

            _, mask = clip_lc.remove_outliers(return_mask=True)
            inverse_mask = np.logical_not(mask)
        ###remove outliers
        while True:
            clip_lc = out_transit.normalize().copy()
            _, mask = out_transit.remove_outliers(return_mask=True)
            inverse_mask = np.logical_not(mask)
            if np.all(inverse_mask) == True:
                print('after clip length: ', len(each_lc.flux))
                each_lc.normalize().errorbar()
                plt.errorbar(out_transit.time.value, out_transit.flux.value, yerr=out_transit.flux_err.value, label='fit_model')
                plt.legend()
                plt.savefig(f'{homedir}/fitting_result/figure/each_lc/{TOInumber}_{datetime.datetime.now().strftime("%y%m%d")}_{str(i)}.png', header=False, index=False)
                #plt.show()
                plt.close()
                break
            else:
                print('cliped:', len(out_transit.flux.value)-len(out_transit[~mask].flux.value))
                out_transit = out_transit[~mask]
        """


        ###transit fitting and clip outliers
        ###params setting
        each_lc = lc[(lc['time'].value > transit_start) & (lc['time'].value < transit_end)]

        ###ignore the case of no bins around transit.
        if len(each_lc) == 0:
            print('no data around transit.')
            continue
        elif np.all(np.isnan(each_lc.flux.value)):
            print('all nan data.')
            continue
        else:
            pass

        '''
        #nanをreplaceする
        nan_index = np.where(np.isnan(each_lc.flux.value))[0].tolist()
        for index in nan_index:
            index_dic = dict(zip(np.where(~(np.isnan(each_lc.flux.value)))[0],  np.abs(np.where(~(np.isnan(each_lc.flux.value)))[0] - index)))
            index_dic_sorted = sorted(index_dic.items(), key=lambda x:x[1])
            replace_flux = (each_lc.flux[index_dic_sorted[0][0]] + each_lc.flux[index_dic_sorted[1][0]]) / 2
            replace_flux_var = np.var(np.array([each_lc.flux[index_dic_sorted[0][0]].value, each_lc.flux[index_dic_sorted[1][0]].value]))
            each_lc.flux[index] = replace_flux + np.random.normal(loc=replace_flux, scale=replace_flux_var)
            replace_flux_err = (each_lc.flux_err[index_dic_sorted[0][0]] + each_lc.flux_err[index_dic_sorted[1][0]]) / 2
            replace_flux_err_var = np.var(np.array([each_lc.flux_err[index_dic_sorted[0][0]].value, each_lc.flux_err[index_dic_sorted[1][0]].value]))
            each_lc.flux_err[index] = replace_flux_err + np.random.normal(loc=replace_flux, scale=replace_flux_err_var)
            '''
        #nanをカットする
        not_nan_index = np.where(~np.isnan(each_lc.flux.value))[0].tolist()
        each_lc = each_lc[not_nan_index]


        noringnames = ["t0", "per", "rp", "a", "inc", "ecc", "w", "q1", "q2"]
        #values = [0.0, 4.0, 0.08, 8.0, 83.0, 0.0, 90.0, 0.2, 0.2]
        if np.isnan(rp/rs):
            values = [mid_transit+period*i, period, 0.08, 8.0, 83.0, 0.0, 90.0, 0.2, 0.2]
        else:
            values = [mid_transit+period*i, period, rp/rs, 8.0, 83.0, 0.0, 90.0, 0.2, 0.2]

        mins = [-0.5, 4.0, 0.03, 4, 80, 0, 90, 0.0, 0.0]
        maxes = [0.5, 4.0, 0.2, 20, 110, 0, 90, 1.0, 1.0]
        #vary_flags = [True, False, True, True, True, False, False, True, True]
        vary_flags = [False, False, True, True, True, False, False, True, True]
        no_ring_params = set_params_lm(noringnames, values, mins, maxes, vary_flags)


        while True:
            try:
                out = lmfit.minimize(no_ring_residual_transitfit,no_ring_params,args=(each_lc.time.value, each_lc.flux.value, each_lc.flux_err.value, noringnames),max_nfev=1000)
            except ValueError:
                print('cant fiting')
                import pdb; pdb.set_trace()
            flux_model = no_ring_model_transitfit_from_lmparams(out.params, each_lc.time.value, noringnames)
            clip_lc = each_lc.normalize().copy()
            clip_lc.flux = np.sqrt(np.square(flux_model - clip_lc.flux))
            _, mask = clip_lc.remove_outliers(return_mask=True)
            inverse_mask = np.logical_not(mask)
            if np.all(inverse_mask) == True:
                print('after clip length: ', len(each_lc.flux))
                each_lc.errorbar()
                plt.plot(each_lc.time.value, flux_model, label='fit_model', color='black')
                plt.axvline(x=mid_transit, color='blue')
                plt.legend()
                #plt.savefig(f'{homedir}/fitting_result/figure/each_lc/{TOInumber}_{str(i)}_{chi_square:.3f}.png', header=False, index=False)
                plt.show()
                import pdb; pdb.set_trace()
                plt.close()
                #each_lc = clip_lc
                break
            else:
                print('cliped:', len(each_lc.flux.value)-len(each_lc[~mask].flux.value))
                each_lc = clip_lc[~mask]

        ###curve fiting
        #out_transit = each_lc[(each_lc['time'].value < (transit_time+period*i)-(duration/2)) | (each_lc['time'].value > (transit_time+period*i)+(duration/2))]
        out_transit = each_lc[(each_lc['time'].value < mid_transit-(duration/2)) | (each_lc['time'].value > mid_transit+(duration/2))]
        model = lmfit.models.PolynomialModel()
        poly_params = model.make_params(c0=1, c1=0, c2=0, c3=0, c4=0, c5=0, c6=0, c7=0)
        result = model.fit(out_transit.flux.value, poly_params, x=out_transit.time.value)
        result.plot()
        #plt.savefig(f'{homedir}/fitting_result/curvefit_figure/{TOInumber}_{i}.png')
        plt.show()
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
        print('before clip length: ', len(each_lc.flux))
        while True:
            _, mask = each_lc.remove_outliers(return_mask=True)
            inverse_mask = np.logical_not(mask)
            if np.all(inverse_mask) == True:
                print('after clip length: ', len(each_lc.flux))
                #each_lc.errorbar()
                #plt.errorbar(each_lc.time.value, each_lc.flux.value, yerr=each_lc.flux_err.value, label='fit_model')
                #plt.legend()
                #plt.savefig(f'{homedir}/fitting_result/figure//{TOInumber}_{str(i)}_{datetime.datetime.now().strftime("%y%m%d")}.png', header=False, index=False)
                #plt.show()
                #plt.close()
                break
            else:
                print('cliped:', len(each_lc.flux.value)-len(each_lc[~mask].flux.value))
                each_lc = each_lc[~mask]
        each_lc_df = each_lc.to_pandas()
        lc_list.append(each_lc_df)
    return lc_list

def folding_each_lc(lc_list, period, transit_time):
    lc = pd.concat(lc_list)
    lc = lc.reset_index()
    lc = Table.from_pandas(lc)
    lc = lk.LightCurve(data=lc)
    lc = lk.LightCurve(time=lc['time'], flux=lc['flux'], flux_err=lc['flux_err'])
    lc = lc.normalize()
    print('total length: ', len(lc))
    return lc.fold(period=period, epoch_time=transit_time)

#if __name__ ==  '__main__':
homedir = '/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani'

sn_df = pd.read_csv(f'{homedir}/toi_overSN100.csv')
TIClist = sn_df['TIC']
params_df = pd.read_excel(f'{homedir}/TOI_parameters.xlsx')
#params_df[params_df['TESS Input Catalog ID']==TIClist[0]].T

for TIC in TIClist:
    param_df = params_df[params_df['TESS Input Catalog ID'] == TIC]
    #tpf = lk.search_targetpixelfile('TIC {}'.format(TIC), mission='TESS', cadence="short").download()
    while True:
        try:
            search_result = lk.search_lightcurve(f'TIC {TIC}', mission='TESS', cadence="short", author='SPOC')
        except HTTPError:
            priont('HTTPError, retry.')
        else:
            break

    #import pdb; pdb.set_trace()
    lc_collection = search_result.download_all()
    try:
        lc_collection.plot()
        #plt.savefig(f'{homedir}/lc_collection/TIC{TIC}.png')
        plt.close()

    except AttributeError:
        #with open('error_tic.dat', 'a') as f:
            #f.write(str(TIC) + '\n')
        continue


    for index, item in param_df.iterrows():
        lc = lc_collection.stitch().flatten() #initialize lc

        duration = item['Planet Transit Duration Value [hours]'] / 24
        period = item['Planet Orbital Period Value [days]']
        transit_time = item['Planet Transit Midpoint Value [BJD]'] - 2457000.0 #translate BTJD
        TOInumber = 'TOI' + str(item["TESS Object of Interest"])
        rp = item['Planet Radius Value [R_Earth]'] * 0.00916794 #translate to Rsun
        rs = item['Stellar Radius Value [R_Sun]']

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
        TOInumber = 'TOI' + str(item["TESS Object of Interest"])

        ###もしもどれかのパラメータがnanだったらそのTIC or TOIを記録して、処理はスキップする。
        pdf = pd.Series([duration, period, transit_time], index=['duration', 'period', 'transit_time'])
        if np.sum(pdf.isnull()) != 0:
            with open('error_tic.dat', 'a') as f:
                f.write(f'nan {pdf[pdf.isnull()].index.tolist()}!:{str(TIC)}+ "\n"')
            continue
        print('analysing: ', TOInumber)
        print('judging whether or not transit is included in the data...')
        time.sleep(1)
        #ターゲットの惑星の信号がデータに影響を与えているか判断
        _, contain_transit, transit_time_list = clip_transit_hoge(lc, duration, period, transit_time, clip_transit=False)

        #ターゲットの惑星の信号がデータに影響を与えていないなら処理を中断する
        if contain_transit == 1:
            '''
            #lc.scatter()
            fig, ax = plt.subplots(figsize =(18, 9))
            for transit in transit_time_list:
                #plt.axvline(x=transit, ymax=np.max(lc.flux.value), ymin=np.min(lc.flux.value), color='red')
                ax.axvline(x=transit, color='blue')
            aaa=lc[lc.time.value < 2000]
            aaa.scatter(ax=ax)
            ax.set_xlim(1325,1355)
            plt.show()
            import pdb; pdb.set_trace()
            plt.savefig(f'{homedir}/check_transit_timing/TIC{TIC}.png')
            #plt.show()
            plt.close()
            '''
            pass

        else:
            print('no transit in data: ', TOInumber)
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
        print('preprocessing...')
        time.sleep(1)
        lc_list = preprocess_each_lc(lc, duration, period, transit_time_list, TOInumber)
        try:
            print('folding...')
            time.sleep(1)
            folded_lc = folding_each_lc(lc_list, period, transit_time)
        except ValueError:
            print('no transit!')
            with open('error_tic.dat', 'a') as f:
                f.write('no transit!: ' + 'str(TIC)' + '\n')
            continue
        folded_lc.scatter()
        #plt.savefig(f'/Users/u_tsubasa/Dropbox/ring_planet_research/folded_lc/figure/{TOInumber}.png')
        plt.show()
        plt.close()
        #folded_lc.write(f'/Users/u_tsubasa/Dropbox/ring_planet_research/folded_lc/data/{TOInumber}.csv')
    #import pdb; pdb.set_trace()

    """
    lc_list = preprocess_each_lc(lc, duration, period, transit_time)
    folded_lc = folding_each_lc(lc_list)
    folded_lc.errorbar()
    plt.show()
    """
