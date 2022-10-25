# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import svd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import lmfit
import lightkurve as lk
import warnings
import c_compile_ring
import batman
from astropy.io import ascii
import os
import sys
import scipy.stats
from scipy import integrate
from concurrent import futures
warnings.filterwarnings('ignore')


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
def ring_model(t, pdic, mcmc_pvalues=None):
    if mcmc_pvalues is None:
        pass
    else:
        for i, param in enumerate(mcmc_params):
            #print(i, v[i])
            pdic[param] = mcmc_pvalues[i]

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
    times = np.array(t)
    #print(pars)
    #print(c_compile_ring.getflux(times, pars, len(times)))
    model_flux = np.array(c_compile_ring.getflux(times, pars, len(times)))*(norm + norm2*(times-t0) + norm3*(times-t0)**2)
    model_flux = np.nan_to_num(model_flux)
    return model_flux

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

def plot_ring(ring_res, rp_rs, rin_rp, rout_rin, b, theta, phi, file_name):
    """
        plotter for rings

        Params:
            rp_rs (float): ratio of planetary raidus to stellar radius
            rp_rin (float):ratio of inner ring radius to planetary raidus
            rout_rin (float):ratio of outer ring radius to inner ring radius
            b (float):impact parameter
            theta (float): ring angle 1 (radian)
            phi (float): ring angle 2 (radian)
        init values:
            rp_rs=0.1, rin_rp=1.3, rout_rin=2, b=0.8, theta=30 * 3.14/180.0, phi = 30 * 3.14/180.0, file_name = "test.pdf"
        Returns:
            None:
    """

    #ring outer & innter radii
    R_in = rp_rs * rin_rp
    R_out = rp_rs * rin_rp * rout_rin

    ## calculte of ellipse of rings
    a0 =1
    b0 = - np.sin(phi) /np.tan(theta)
    c0 = (np.sin(phi)**2)/(np.tan(theta)**2) + ((np.cos(phi)**2)/(np.sin(theta)**2))
    A = np.array([[a0, b0],[b0,c0]])
    u, s, vh = svd(A)
    angle = (np.arctan2(u[0][0], u[0][1]))*180/np.pi
    major = s[0]
    major_to_minor = np.sqrt((s[1]/s[0]))

    #plotter
    fig = plt.figure(figsize = (8,8))
    ax = plt.axes()
    c = patches.Circle(xy=(0, 0), radius=1.0, fc='orange', ec='orange')
    c2 = patches.Circle(xy=(0, -b), radius=rp_rs, fc='k', ec='k')
    e = patches.Ellipse(xy=(0, -b), width=2 * R_in, height=2 * R_in * major_to_minor, angle = angle, fc='none', ec='r')
    e2 = patches.Ellipse(xy=(0, -b), width=2 * R_out, height=2 * R_out * major_to_minor, angle = angle, fc='none', ec='r')
    ax.add_patch(c)
    ax.add_patch(c2)
    ax.add_patch(e)
    ax.add_patch(e2)
    ax.set_title(f'chisq={str(ring_res.redchi)[:6]}')
    plt.axis('scaled')
    ax.set_aspect('equal')
    os.makedirs(f'{lmfit_dir}/illustration/{TOInumber}', exist_ok=True)
    #os.makedirs(f'./simulation/illustration/{TOInumber}', exist_ok=True)
    plt.savefig(f'{lmfit_dir}/illustration/{TOInumber}/{file_name}', bbox_inches="tight")
    #plt.show()

def calc_ring_res(m, no_ring_res, t, flux_data, flux_err_data, TOInumber):
    names = ["q1", "q2", "t0", "porb", "rp_rs", "a_rs",
            "b", "norm", "theta", "phi", "tau", "r_in",
            "r_out", "norm2", "norm3", "ecosw", "esinw"]
    values = [no_ring_res.params.valuesdict()['q1'], no_ring_res.params.valuesdict()['q2'], no_ring_res.params.valuesdict()['t0'], period, no_ring_res.params.valuesdict()['rp'], no_ring_res.params.valuesdict()['a'],
                no_ring_res.params.valuesdict()['b'], 1, np.random.uniform(1e-5,np.pi-1e-5), np.random.uniform(0.0,np.pi), 1, np.random.uniform(1.01,2.44),
                np.random.uniform(1.02,2.44), 0.0, 0.0, 0.0, 0.0]

    saturnlike_values = [0.0, 0.7, 0.0, 4.0, 0.18, 10.7,
                1, 1, np.pi/6.74, 0, 1, 1.53,
                1.95, 0.0, 0.0, 0.0, 0.0]

    mins = [0.0, 0.0, -0.1, 0.0, 0.01, 1.0,
            0.0, 0.9, 0.0, 0.0, 0.0, 1.00,
            1.01, -0.1, -0.1, 0.0, 0.0]

    maxes = [1.0, 1.0, 0.1, 100.0, 0.5, 100.0,
                1.0, 1.1, np.pi, np.pi, 1.0, 2.45,
                2.45, 0.1, 0.1, 0.0, 0.0]

    vary_flags = [True, True, True, False, True, True,
                    True, False, True, True, False, True,
                    True, False, False, False, False]

    params = set_params_lm(names, values, mins, maxes, vary_flags)
    params_df = pd.DataFrame(list(zip(values, saturnlike_values, mins, maxes)), columns=['values', 'saturnlike_values', 'mins', 'maxes'], index=names)
    vary_dic = dict(zip(names, vary_flags))
    params_df = params_df.join(pd.DataFrame.from_dict(vary_dic, orient='index', columns=['vary_flags']))
    pdic = params_df['values'].to_dict()

    ring_res = lmfit.minimize(ring_transitfit, params, args=(t, flux_data, flux_err_data.mean(), names), max_nfev=1000)
    if ring_res.params['r_out'].stderr != None and np.isfinite(ring_res.params['r_out'].stderr):
        F_obs = ( (no_ring_res.chisqr-ring_res.chisqr)/ (ring_res.nvarys-no_ring_res.nvarys) ) / ( ring_res.chisqr/(ring_res.ndata-ring_res.nvarys-1) )
        p_value = 1 - integrate.quad( lambda x:scipy.stats.f.pdf(x, ring_res.ndata-ring_res.nvarys-1, ring_res.nvarys-no_ring_res.nvarys), 0, F_obs )[0]
        #best_ring_res_dict[F_obs] = ring_res

        ###csvにfitting parametersを書き出し###
        input_df = pd.DataFrame.from_dict(params.valuesdict(), orient="index",columns=["input_value"])
        output_df = pd.DataFrame.from_dict(ring_res.params.valuesdict(), orient="index",columns=["output_value"])
        input_df=input_df.applymap(lambda x: '{:.6f}'.format(x))
        output_df=output_df.applymap(lambda x: '{:.6f}'.format(x))
        result_df = input_df.join((output_df, pd.Series(vary_flags, index=names, name='vary_flags')))
        os.makedirs(f'{lmfit_dir}/fit_p_data/ring_model/{TOInumber}', exist_ok=True)
        #os.makedirs(f'./simulation/fit_p_data/{TOInumber}', exist_ok=True)
        result_df.to_csv(f'{lmfit_dir}/fit_p_data/ring_model/{TOInumber}/{TOInumber}_{F_obs:.0f}_{m}.csv', header=True, index=False)

        plot_ring(ring_res, rp_rs=ring_res.params['rp_rs'].value, rin_rp=ring_res.params['r_in'].value, rout_rin=ring_res.params['r_out'].value, b=ring_res.params['b'].value, theta=ring_res.params['theta'].value, phi=ring_res.params['phi'].value, file_name = f"{TOInumber}_{F_obs:.0f}_{m}.pdf")

        fig = plt.figure()
        ax_lc = fig.add_subplot(2,1,1) #for plotting transit model and data
        ax_re = fig.add_subplot(2,1,2) #for plotting residuals
        ring_flux_model = ring_transitfit(ring_res.params, t, flux_data, flux_err_data, names, return_model=True)
        noring_flux_model = no_ring_transitfit(no_ring_res.params, t, flux_data, flux_err_data, noringnames, return_model=True)
        binned_lc.errorbar(ax=ax_lc)
        ax_lc.plot(t, ring_flux_model, label='Model w/ ring', color='blue')
        ax_lc.plot(t, noring_flux_model, label='Model w/o ring', color='red')
        residuals_ring = binned_lc - ring_flux_model
        residuals_no_ring = binned_lc - noring_flux_model
        residuals_ring.plot(ax=ax_re, color='blue', alpha=0.3,  marker='.', zorder=1)
        residuals_no_ring.plot(ax=ax_re, color='red', alpha=0.3,  marker='.', zorder=1)
        ax_re.plot(t, np.zeros(len(t)), color='black', zorder=2)
        ax_lc.legend()
        #ax_lc.set_title(f'w/ chisq:{ring_res.chisqr:.0f}/{ring_res.nfree:.0f} w/o chisq:{no_ring_res.chisqr:.0f}/{no_ring_res.nfree:.0f}')
        ax_lc.set_title(f'{TOInumber}, p={p_value:.2e}')
        plt.tight_layout()
        os.makedirs(f'{lmfit_dir}/transit_fit/{TOInumber}', exist_ok=True)
        #os.makedirs(f'./simulation/transit_fit/{TOInumber}', exist_ok=True)
        plt.savefig(f'{lmfit_dir}/transit_fit/{TOInumber}/{TOInumber}_{F_obs:.0f}_{m}.pdf')
        #plt.show()
        plt.close()
        
        return {"res":ring_res, "F_obs":F_obs}
    else:
        return {"res":ring_res, "F_obs":-999}
    
def calc_bin_std(lc, TOInumber):
    std_list =[]
    std_err_list =[]
    phase_list = []
    min_list = lc.bin(bins=500).remove_nans().time.value
    for i in range(len(min_list)):
        try:
            part_lc = lc[lc.time.value > min_list[i]]
            part_lc = part_lc[part_lc.time.value < min_list[i+1]]
            if len(part_lc) == 0:
                continue
            std_list.append(part_lc.flux.std())
            std_err_list.append(part_lc.flux_err.std())
            phase_list.append(part_lc.time[0].value)    
        except IndexError:
            pass
    #print(len(std_list))
    #print(len(phase_list))
    fig = plt.figure()
    ax2 = fig.add_subplot(2,1,1) #for plotting transit model and data
    ax1 = fig.add_subplot(2,1,2) #for plotting residuals
    lc.errorbar(ax=ax1, color='black', marker='.', zorder=1, label='data')
    ax2.errorbar(x=phase_list, y=std_list, yerr=std_err_list, fmt='.k')
    #ax2.set_xlabel('orbital phase')
    ax2.set_ylabel('Flux std')
    ax2.set_title(TOInumber)
    plt.savefig(f'/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/fitting_result/figure/bin_std/{TOInumber}.png')
    #plt.show()
    d = {'phase':phase_list, 'bin_std':std_list}
    pd.DataFrame(data=d).to_csv(f'/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/fitting_result/data/bin_std/{TOInumber}.csv')
    plt.close()

#csvfile = './folded_lc_data/TOI2403.01.csv'
#done_TOIlist = os.listdir('./lmfit_result/transit_fit') #ダブリ解析防止
#oridf = pd.read_csv('./exofop_tess_tois.csv')
oridf = pd.read_csv('./exofop_tess_tois_2022-09-13.csv')
lmfit_dir='/Users/u_tsubasa/Dropbox/ring_planet_research/lmfit_result'
df = oridf[oridf['Planet SNR']>100]

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

duration_ng = [129.01, 182.01, 1059.01, 1182.01, 1425.01, 1455.01] 
trend_ng = [1069.01, 1092.01, 1141.01, 1163.01, 1198.01, 1270.01, 1299.01, 1385.01, 1454.01, 1455.01, 1647.01, 1796.01,]
'''
df['Rjup'] = df['Planet Radius (R_Earth)']/11.209
plt.scatter(x=df['Period (days)'], y = df['Rjup'], color='k')
plt.xlabel('Orbital Period(day)')
plt.ylabel(r'Planet Radius ($R_{J}$)')
plt.gca().yaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
plt.gca().xaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
plt.xscale('log')
plt.minorticks_on()
plt.show()
import pdb;pdb.set_trace()

done_list = os.listdir('/mwork2/umetanitb/research_umetani/lmfit/transit_fit/')
done_list = [s for s in done_list if 'TOI' in s]
done_list = [s.lstrip('TOI') for s in done_list ]
#done_list = [float(s.strip('.pdf')) for s in done_list]
done_list = [float(s) for s in done_list]
'''
df['TOI'] = df['TOI'].astype(str)
df = df.sort_values('Planet SNR', ascending=False)
df = df.set_index(['TOI'])
#df = df.drop(index=done_list, errors='ignore')
df = df.drop(index=no_data_list, errors='ignore')
#df = df.drop(index=multiplanet_list, errors='ignore')
#df = df.drop(index=startrend_list, errors='ignore')
#df = df.drop(index=flare_list, errors='ignore')
df = df.drop(index=two_epoch_list, errors='ignore')
df = df.drop(index=no_signal_list, errors='ignore')
#df = df.drop(index=duration_ng, errors='ignore')
#df = df.drop(index=trend_ng, errors='ignore')
df = df.reset_index()


TOIlist = df['TOI']
#for TOI in [495.01]:

#TOI = TOIlist[10]
#for TOI in [129.01,]:
for TOI in TOIlist:
    TOI = str(TOI)
    print(TOI)
    TOInumber = 'TOI' + TOI
    param_df = df[df['TOI'] == TOI]
    '''
    ###parameters setting###
    #TOInumber = 'TOI' + str(param_df['TOI'].values[0])
    try:
        with open(f'/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/fitting_result/data/folded_lc/4poly/calc_t0/fit_statistics/{TOInumber}/{TOInumber}_folded.txt', 'r') as f:
            #durationline = f.readlines()[-5:].split(' ')[-1]
            durationline, _, _, periodline, _ = f.readlines()[-5:]
            durationline = durationline.split(' ')[-1]
            duration = np.float(durationline)
            periodline = periodline.split(' ')[-1]
            period = np.float(periodline)
    except FileNotFoundError:
        with open('./folded.txt_FileNotFoundError.txt', 'a') as f:
            f.write(TOInumber+'\n')
            duration = param_df['Duration (hours)'].values[0] / 24
            period = param_df['Period (days)'].values[0]

    rp = param_df['Planet Radius (R_Earth)'].values[0] * 0.00916794 #translate to Rsun
    rs = param_df['Stellar Radius (R_Sun)'].values[0]
    rp_rs = rp/rs
    '''

    #csvfile = f'/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/fitting_result/data/folded_lc/4poly/obs_t0/csv/{TOInumber}.csv'
    csvfile = f'/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/fitting_result/data/folded_lc/20220913/obs_t0/csv/{TOInumber}.csv'
    try:
        folded_table = ascii.read(csvfile)
    except FileNotFoundError:
        #with open('./csvfile_FileNotFoundError.txt', 'a') as f:
            #f.write(TOInumber+',')
        continue
    folded_lc = lk.LightCurve(data=folded_table)
    #folded_lc = folded_lc[(folded_lc.time.value < duration) & (folded_lc.time.value > -duration)]
    calc_bin_std(folded_lc, TOInumber)
    continue
    binned_lc = folded_lc.bin(bins=500).remove_nans()
    
    t = binned_lc.time.value
    flux_data = binned_lc.flux.value
    flux_err_data = binned_lc.flux_err.value

    if np.sum(np.isnan([t, flux_data, flux_err_data])) != 0:
        with open('./NaN_values_detected.txt', 'a') as f:
            f.write(TOInumber+',')
            continue
    ###no ring model fitting by minimizing chi_square###
    best_res_dict = {}
    for n in range(50):
        noringnames = ["t0", "per", "rp", "a", "b", "ecc", "w", "q1", "q2"]
        #values = [0.0, 4.0, 0.08, 8.0, 83.0, 0.0, 90.0, 0.2, 0.2]
        #noringvalues = [0, period, rp_rs, a_rs, 83.0, 0.0, 90.0, 0.2, 0.2]
        if np.isnan(rp_rs):
            noringvalues = [np.random.uniform(-0.05,0.05), period, np.random.uniform(0.05,0.1), np.random.uniform(1,10), np.random.uniform(0.0,0.5), 0, 90.0, np.random.uniform(0.1,0.9), np.random.uniform(0.1,0.9)]
        else:
            noringvalues = [np.random.uniform(-0.05,0.05), period, rp_rs, np.random.uniform(1,10), np.random.uniform(0.0,0.5), 0, 90.0, np.random.uniform(0.1,0.9), np.random.uniform(0.1,0.9)]
        noringmins = [-0.2, period*0.8, 0.01, 1, 0, 0, 90, 0.0, 0.0]
        noringmaxes = [0.2, period*1.2, 0.5, 100, 1.0, 0.8, 90, 1.0, 1.0]
        noringvary_flags = [True, False, True, True, True, False, False, True, True]
        no_ring_params = set_params_lm(noringnames, noringvalues, noringmins, noringmaxes, noringvary_flags)
        no_ring_res = lmfit.minimize(no_ring_transitfit, no_ring_params, args=(t, flux_data, flux_err_data, noringnames), max_nfev=1000)
        if no_ring_res.params['t0'].stderr != None:
            if np.isfinite(no_ring_res.params['t0'].stderr) and no_ring_res.redchi < 10:
                red_redchi = no_ring_res.redchi-1
                best_res_dict[red_redchi] = no_ring_res
    no_ring_res = sorted(best_res_dict.items())[0][1]
    input_df = pd.DataFrame.from_dict(no_ring_params.valuesdict(), orient="index",columns=["input_value"])
    output_df = pd.DataFrame.from_dict(no_ring_res.params.valuesdict(), orient="index",columns=["output_value"])
    input_df = input_df.applymap(lambda x: '{:.6f}'.format(x))
    output_df = output_df.applymap(lambda x: '{:.6f}'.format(x))
    result_df = input_df.join((output_df, pd.Series(noringvary_flags, index=noringnames, name='vary_flags')))
    os.makedirs(f'{lmfit_dir}/fit_p_data/no_ring_model/{TOInumber}', exist_ok=True)
    result_df.to_csv(f'{lmfit_dir}/fit_p_data/no_ring_model/{TOInumber}/{TOInumber}_{no_ring_res.redchi:.2f}.csv', header=True, index=False)

    ###ring model fitting by minimizing chi_square###
    future_list = []
    #best_ring_res_dict = {}
    with futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_list = [executor.submit(fn=calc_ring_res, 
                                  m=m,
                                  no_ring_res=no_ring_res, 
                                  t=t, 
                                  flux_data=flux_data, 
                                  flux_err_data=flux_err_data, 
                                  TOInumber=TOInumber) for m in range(100)]
        future_res_list = [f.result() for f in future_list]
    print(future_res_list)
    ring_res = list(sorted(future_res_list, key=lambda x: x['F_obs'], reverse=True)[0].values())[0]
    #ring_res = sorted(best_ring_res_dict.items(), reverse=True)[0][1]
    F_obs = ( (no_ring_res.chisqr-ring_res.chisqr)/ (ring_res.nvarys-no_ring_res.nvarys) ) / ( ring_res.chisqr/(ring_res.ndata-ring_res.nvarys-1) )
    p_value = 1 - integrate.quad( lambda x:scipy.stats.f.pdf(x, ring_res.ndata-ring_res.nvarys-1, ring_res.nvarys-no_ring_res.nvarys), 0, F_obs )[0]
    with open(f'{lmfit_dir}/fit_report/{TOInumber}.txt', 'a') as fi:
        print('no ring transit fit report:\n', file = fi)
        print(lmfit.fit_report(no_ring_res), file = fi)
        print('ring transit fit report:\n', file = fi)
        print(lmfit.fit_report(ring_res), file = fi)
        print(f'\nF_obs: {F_obs}', file = fi)
        print(f'\np_value: {p_value}', file = fi)
    
