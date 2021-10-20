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


def preprocess_each_lc(lc, duration, period, transit_time, TOInumber):
    folded_lc = lc.fold(period=period , epoch_time=transit_time)
    ###nanをカットする
    not_nan_index = np.where(~np.isnan(folded_lc.flux.value))[0].tolist()
    folded_lc = folded_lc[not_nan_index]

    ###トランジットエポックの検出
    epoch_all_time = ( (folded_lc.time_original.value - transit_time) + 0.5*period ) / period
    epoch_all= np.array(epoch_all_time, dtype = int)
    epoch_all_list = list(set(epoch_all))
    epoch_all_list = np.sort(epoch_all_list)
    folded_lc.epoch_all = epoch_all
    if len(np.unique(epoch_all)) != len(epoch_all_list):
        print('check: len(np.unique(epoch_all)) != len(epoch_all_list).')
        import pdb; pdb.set_trace()
    print(f'n_transit: {len(np.unique(epoch_all))}')

    ###トランジットフィッティングパラメータの設定
    names = ["t0", "per", "rp", "a", "inc", "ecc", "w", "q1", "q2"]
    if np.isnan(rp/rs):
        values = [0, period, 0.02, 10, 87, 0, 90, 0.3, 0.2]
    else:
        values = [0, period, rp/rs, 10, 87, 0, 90, 0.3, 0.2]
    mins = [-0.5, period*0.99, 0.001, 1, 70, 0, 90, 0.0, 0.0]
    maxes = [0.5, period*1.01, 0.2, 100, 110, 0, 90, 1.0, 1.0]
    vary_flags = [True, True, True, True, True, False, False,False , False]
    params = set_params_lm(names, values, mins, maxes, vary_flags)

    ###値を格納するリスト
    t0dict = {}
    perioddict = {}
    time_now_arr = []
    each_lc_list = []
    outliers_list = []

    ###それぞれのトランジットエポックごとにトランジットフィッティングと外れ値除去、カーブフィッティング
    for i, epoch_now in enumerate(epoch_all_list):
        print(epoch_now)
        flag = folded_lc.epoch_all == epoch_now
        each_lc = folded_lc[flag]

        if len(each_lc[(each_lc.time < 0.01) & (each_lc.time > -0.01)]) == 0:
            each_lc.errorbar()
            plt.title('no data in mid transit')
            plt.savefig(f'{homedir}/fitting_result/figure/error_lc/{TOInumber}_{str(i)}.png', header=False, index=False)
            plt.close()
            continue

        print(np.min(each_lc.time_original.value), np.max(each_lc.time_original.value))
        '''
        if np.min(each_lc.time_original.value) < 2000:
            values = [0.0, period, 0.1, 10, 87, 0, 90, 0.3, 0.2]
        else:
            values = [0.0, period, 0.1, 10, 87, 0, 90, 0.3, 0.2]
        '''
        params = set_params_lm(names, values, mins, maxes, vary_flags)

        while True:
            ### transit fitting
            try:
                #flag_time = np.abs(each_lc.time.value)<1.0
                #each_lc = each_lc[flag_time]
                time = each_lc.time.value
                flux = each_lc.flux.value
                flux_err = each_lc.flux_err.value
                out = lmfit.minimize(no_ring_residual_transitfit, params, args=(time, flux, flux_err, names))
            except TypeError:
                print('TypeError: out')
                import pdb; pdb.set_trace()
                continue
            except ValueError:
                print('cant fiting')
                import pdb; pdb.set_trace()

            ### remove outliers
            try:
                if np.isfinite(out.params["t0"].stderr):
                    #print(out.params.pretty_print())
                    time_now_arr.append(0.5 * np.min(each_lc.time_original.value) + 0.5* np.max(each_lc.time_original.value))
                    flux_model = no_ring_model_transitfit_from_lmparams(out.params, time, names)

                    clip_lc = each_lc.copy()
                    clip_lc.flux = np.sqrt(np.square(flux_model - clip_lc.flux))
                    _, mask = clip_lc.remove_outliers(return_mask=True)
                    inverse_mask = np.logical_not(mask)
                    ax = plt.subplot(1,1,1)

                    if np.all(inverse_mask) == True:
                        #print(f'after clip length: {len(each_lc.flux)}')
                        each_lc.errorbar(ax=ax, color='black')
                        ax.plot(time,flux_model, label='fit_model', color='blue')
                        ax.legend()
                        #ax.set_xlim(-1, 1)
                        #plt.savefig(f'{homedir}/fitting_result/figure/each_lc/{TOInumber}_{str(i)}_{int(chi_square)}.png', header=False, index=False)
                        #plt.show()
                        plt.close()
                        t0dict[epoch_now] = [transit_time+(period*epoch_now)+out.params["t0"].value, out.params["t0"].stderr]
                        #t0dict[i] = [out.params["t0"].value, out.params["t0"].stderr]
                        perioddict[i] = [out.params["per"].value, out.params["per"].stderr]
                        #each_lc = clip_lc
                        break
                    else:
                        print('removed bins:', len(each_lc[mask]))
                        outliers_list.append(each_lc[mask])
                        each_lc[mask].errorbar(ax=ax, color='red', label='outliers')
                        each_lc = each_lc[~mask]
            except TypeError:
                each_lc.errorbar()
                plt.xlim(-1, 1)
                plt.savefig(f'{homedir}/fitting_result/figure/error_lc/{TOInumber}_{str(i)}.png', header=False, index=False)
                plt.close()
                break

        ###curve fiting
        #out_transit = each_lc[(each_lc['time'].value < (transit_time+period*i)-(duration/2)) | (each_lc['time'].value > (transit_time+period*i)+(duration/2))]
        out_transit = each_lc[(each_lc['time'].value < out.params["t0"].value - (duration/2)) | (each_lc['time'].value > out.params["t0"].value + (duration/2))]
        model = lmfit.models.PolynomialModel()
        poly_params = model.make_params(c0=1, c1=0, c2=0, c3=0, c4=0, c5=0, c6=0, c7=0)
        result = model.fit(out_transit.flux.value, poly_params, x=out_transit.time.value)
        result.plot()
        #plt.savefig(f'{homedir}/fitting_result/curvefit_figure/{TOInumber}_{i}.png')
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

    cleaned_lc = vstack(each_lc_list)
    outliers = vstack(outliers_list)


    t0df = pd.DataFrame.from_dict(t0dict, orient='index', columns=['t0', 't0err'])
    import pdb; pdb.set_trace()
    plt.errorbar(x=t0df.index.values, y=t0df['t0'],yerr=t0df['t0err'], fmt='.k')
    res = np.polyfit(t0df.index.values, t0df['t0'],1)
    plt.plot(t0df.index.values, np.poly1d(res)(t0df.index.values))
    plt.show()
    '''
    perioddf = pd.DataFrame.from_dict(perioddict, orient='index', columns=['period', 'perioderr'])
    plt.scatter(x=perioddf.index.values, y=perioddf['period'])
    plt.show()
    '''

    import pdb; pdb.set_trace()
    return cleaned_lc, outliers

def folding_each_lc(lc_list, period, transit_time):
    #binned_lc = folded_lc.bin(time_bin_size=duration/20)
    #binned_lc.scatter()
    #plt.show()
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
        lc_list = preprocess_each_lc(lc, duration, period, transit_time, TOInumber)
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
