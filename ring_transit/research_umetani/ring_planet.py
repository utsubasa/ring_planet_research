# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import lmfit
import lightkurve as lk
from fit_model import model
import warnings
import c_compile_ring
import batman
import datetime
import time
#import emcee
import corner
from multiprocessing import Pool
#from lightkurve import search_targetpixelfile
from scipy import signal
from astropy.io import ascii
import glob
import os
import sys
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
def ring_model(t, pdic, mcmc_pvalues=None):
    if mcmc_pvalues is None:
        pass
    else:
        ##import pdb; pdb.set_trace()
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
    print(pars)
    #print(c_compile_ring.getflux(times, pars, len(times)))
    model_flux = np.array(c_compile_ring.getflux(times, pars, len(times)))*(norm + norm2*(times-t0) + norm3*(times-t0)**2)
    model_flux = np.nan_to_num(model_flux)
    return model_flux

#リングありモデルをfitting
def ring_residual_transitfit(params, x, data, eps_data, names):
    model = ring_model(x, params.valuesdict())
    chi_square = np.sum(((data-model)/eps_data)**2)
    #print(params)
    #print(chi_square)
    #print(np.max(((data-model)/eps_data)**2))

    return (data-model) / eps_data

#リングなしモデルをfitting
def no_ring_residual_transitfit(params, x, data, eps_data, names):
    #global chi_square
    params_batman = set_params_batman(params, names)
    m = batman.TransitModel(params_batman, x)    #initializes model
    model = m.light_curve(params_batman)         #calculates light curve
    chi_square = np.sum(((data-model)/eps_data)**2)
    #if chi_square < 290:
        #import pdb; pdb.set_trace()
    #print(params)
    #print(chi_square)
    return (data-model)/eps_data

def ring_model_transitfit_from_lmparams(params, x):
    model = ring_model(x, params.valuesdict())         #calculates light curve
    return model

def no_ring_model_transitfit_from_lmparams(params, x, names):
    params_batman = set_params_batman(params, names)
    m = batman.TransitModel(params_batman, x)    #initializes model
    model = m.light_curve(params_batman)
    return model

def lnlike(mcmc_pvalues, t, y, yerr):
    return -0.5 * np.sum(((y-ring_model(t, pdic, mcmc_pvalues))/yerr) ** 2)

def log_prior(mcmc_pvalues, mcmc_params):
    #p_dict = dict(zip(mcmc_params, p))
    #rp_rs, theta, phi, r_in = p
    for i, param in enumerate(mcmc_params):
        if df_for_mcmc['mins'][param] <= mcmc_pvalues[i] <= df_for_mcmc['maxes'][param]:
            pass
        else:
            return -np.inf
        if param =='r_out' and mcmc_pvalues[i] > 3.0:
            return -np.inf
        else:
            pass
    #if 0.0 < theta < np.pi/2 and 0.0 < phi < np.pi/2 and 0.0 < rp_rs < 1 and 1.0 < r_in < 7.0:
    #    return 0.0
    return 0.0

def lnprob(mcmc_pvalues, t, y, yerr, mcmc_params):
    lp = log_prior(mcmc_pvalues, mcmc_params)
    if not np.isfinite(lp):
        return -np.inf
    chi_square = np.sum(((y-ring_model(t, pdic, mcmc_pvalues))/yerr)**2)
    print(chi_square)

    return lp + lnlike(mcmc_pvalues, t, y, yerr)

def folding_each_lc(lc_list):
    lc = pd.concat(lc_list[:4])
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
    for n in range(n_transit):
        target_val = transit_time + (period * n)
        mid_transit_row = getNearestRow(lc.time.value, target_val)
        list.append(mid_transit_row)
    return list

def draw_ringplanet(pdic, mcmc_pvalues):
    for i, param in enumerate(mcmc_params):
        pdic[param] = mcmc_pvalues[i]
    phi = pdic['phi']
    theta = pdic['theta']
    Rphi = np.array([[np.cos(phi), -np.sin(phi), 0.], [np.sin(phi), np.cos(phi), 0.], [0.,0.,1.]])
    Rtheta = np.array([[1.,0.,0.], [0., np.cos(theta), -np.sin(theta)], [0., np.sin(theta), np.cos(theta)]])
    R = np.dot(Rphi, Rtheta)
    Router = 0.06
    Rinner = 0.04
    Rsaturn = 0.018

    #rs0 = np.array([[np.cos(phi), np.sin(phi), 0.] for phi in np.linspace(0., 2*np.pi, 100)])
    phivec1 = np.linspace(0., np.pi, 50)
    phivec2 = np.linspace(np.pi,2*np.pi, 50)
    rs1 = np.array([np.cos(phivec1), np.sin(phivec1), np.zeros_like(phivec1)]).T  # first half arc
    rs2 = np.array([np.cos(phivec2), np.sin(phivec2), np.zeros_like(phivec2)]).T  # second half arc
    rs0 = np.concatenate((rs1,rs2),axis=0)  # full arc for Saturn
    rs1 = np.dot(rs1, R.T)  # rotate
    rs2 = np.dot(rs2, R.T)  # rotate

    # draw foreground
    semiring1 = np.concatenate((Router*rs1[:,:2],Rinner*rs1[::-1,:2]),axis=0)
    plt.fill(semiring1[:,0], semiring1[:,1],'blue',edgecolor='none')
    # draw Saturn
    plt.fill(Rsaturn*rs0[:,0], Rsaturn*rs0[:,1],'yellow')
    # draw foreground
    semiring2 = np.concatenate((Router*rs2[:,:2],Rinner*rs2[::-1,:2]),axis=0)
    plt.fill(semiring2[:,0], semiring2[:,1],'blue',edgecolor='none')
    plt.axis('equal')
    plt.show()
"""
phi = 0
theta = 0.01
Rphi = np.array([[np.sin(phi), -np.cos(phi), 0.], [np.cos(phi), np.sin(phi), 0.], [0.,0.,1.]])
Rtheta = np.array([[1.,0.,0.], [0., np.sin(theta), -np.cos(theta)], [0., np.cos(theta), np.sin(theta)]])
R = np.dot(Rphi, Rtheta)
Router = 0.06
Rinner = 0.04
Rsaturn = 0.018

#rs0 = np.array([[np.cos(phi), np.sin(phi), 0.] for phi in np.linspace(0., 2*np.pi, 100)])
phivec1 = np.linspace(0., np.pi, 50)
phivec2 = np.linspace(np.pi,2*np.pi, 50)
rs1 = np.array([np.sin(phivec1), np.cos(phivec1), np.zeros_like(phivec1)]).T  # first half arc
rs2 = np.array([np.sin(phivec2), np.cos(phivec2), np.zeros_like(phivec2)]).T  # second half arc
rs0 = np.concatenate((rs1,rs2),axis=0)  # full arc for Saturn
rs1 = np.dot(rs1, R.T)  # rotate
rs2 = np.dot(rs2, R.T)  # rotate

# draw foreground
semiring1 = np.concatenate((Router*rs1[:,:2],Rinner*rs1[::-1,:2]),axis=0)
plt.fill(semiring1[:,0], semiring1[:,1],'blue',edgecolor='none')
# draw Saturn
plt.fill(Rsaturn*rs0[:,0], Rsaturn*rs0[:,1],'yellow')
# draw foreground
semiring2 = np.concatenate((Router*rs2[:,:2],Rinner*rs2[::-1,:2]),axis=0)
plt.fill(semiring2[:,0], semiring2[:,1],'blue',edgecolor='none')
plt.axis('equal')
plt.show()
import pdb; pdb.set_trace()
"""

#csvfile = './folded_lc_data/TOI2403.01.csv'
df = pd.read_csv('./exofop_tess_tois.csv')
df = df[df['Planet SNR']>100]
df['TOI'] = df['TOI'].astype(str)
#df = df.sort_values('Planet SNR', ascending=False)
for TOI in df['TOI'].values:
    param_df = df[df['TOI'] ==TOI]
    #lm.minimizeのためのparamsのセッティング。これはリングありモデル
    ###parameters setting###
    for index, item in param_df.iterrows():
        TOInumber = 'TOI' + str(item['TOI'])
        try:
            duration = item['Duration (hours)'] / 24
            period = item['Period (days)']
            transit_time = item['Transit Epoch (BJD)'] - 2457000.0 #translate BTJD
            a_rs=4.602
            b=0.9
            rp = item['Planet Radius (R_Earth)'] * 0.00916794 #translate to Rsun
            rs = item['Stellar Radius (R_Sun)']
        except NameError:
            print(f'error:{TOInumber}')
            continue
        #Mp = ?
        rp_rs = rp/rs
        i=87.21 * 0.0175 #radian
        au=0.0376 #Orbit Semi-Major Axis [au]
        au=au*1.496e+13 #cm
        rstar=rs * 6.9634 * 10**10 #Rstar cm
        a_rs=au/rstar
        #rplanet = rp * 6.9634 * 10**10
        #rmin = np.pow(Mp/(4*np.pi/3)*8.87, 1/3)
        #rp_rs_min = rmin/rs
    csvfile = f'./binned_lc_data/{TOInumber}.csv'
    try:
        folded_table = ascii.read(csvfile)
    except FileNotFoundError:
        continue
    folded_lc = lk.LightCurve(data=folded_table)
    folded_lc = folded_lc[(folded_lc.time.value < duration*0.7) & (folded_lc.time.value > -duration*0.7)]
    import astropy.units as u
    binned_lc = folded_lc.bin(time_bin_size=1*u.minute).remove_nans()
    '''
    for file in files:
        try:
            print(file)
            folded_df = pd.read_csv(file, sep=',')
            folded_table = Table.from_pandas(folded_df)
            folded_lc = lk.LightCurve(data=folded_table)
            import astropy.units as u
            binned_lc = folded_lc.bin(time_bin_size=5*u.minute)
            binned_lc.errorbar()
            #plt.show()
        except:
            pass
    #folded_lc = folded_lc.bin(bins=300)
    import pdb; pdb.set_trace()
    '''

    t = binned_lc.time.value
    flux_data = binned_lc.flux.value
    flux_err_data = binned_lc.flux_err.value

    #t = np.linspace(-0.2, 0.2, 300)

    ###ring model fitting by minimizing chi_square###
    best_res_dict = {}
    for n in range(20):
        noringnames = ["t0", "per", "rp", "a", "inc", "ecc", "w", "q1", "q2"]
        #values = [0.0, 4.0, 0.08, 8.0, 83.0, 0.0, 90.0, 0.2, 0.2]
        #noringvalues = [0, period, rp_rs, a_rs, 83.0, 0.0, 90.0, 0.2, 0.2]
        noringvalues = [0, period, rp/rs, np.random.uniform(1.01,20.0), 80.0, 0.5, 90.0, np.random.uniform(0.01,1.0), np.random.uniform(0.01,1.0)]
        noringmins = [-0.1, period*0.9, 0.03, 1, 70, 0.0, 90, 0.0, 0.0]
        noringmaxes = [0.1, period*1.1, 0.2, 20, 110, 1.0, 90, 1.0, 1.0]
        #vary_flags = [True, False, True, True, True, False, False, True, True]
        noringvary_flags = [True, True, True, True, True, True, False, True, True]
        no_ring_params = set_params_lm(noringnames, noringvalues, noringmins, noringmaxes, noringvary_flags)
        #start = time.time()

        no_ring_res = lmfit.minimize(no_ring_residual_transitfit, no_ring_params, args=(t, flux_data, flux_err_data, noringnames), max_nfev=1000)
        best_res_dict[no_ring_res.chisqr] = no_ring_res
        #print(lmfit.fit_report(no_ring_res))
    no_ring_res = sorted(best_res_dict.items())[0][1]
    best_ring_res_dict = {}
    for m in range(20):
        print(m, TOInumber)
        names = ["q1", "q2", "t0", "porb", "rp_rs", "a_rs",
                 "b", "norm", "theta", "phi", "tau", "r_in",
                 "r_out", "norm2", "norm3", "ecosw", "esinw"]
        values = [no_ring_res.params.valuesdict()['q1'], no_ring_res.params.valuesdict()['q2'], no_ring_res.params.valuesdict()['t0'], no_ring_res.params.valuesdict()['per'], no_ring_res.params.valuesdict()['rp'], no_ring_res.params.valuesdict()['a'],
                  b, 1, np.random.uniform(1e-5,np.pi-1e-5), np.random.uniform(1e-5,np.pi-1e-5), 1, np.random.uniform(1.01,3.0),
                  np.random.uniform(1.01,3.0), 0.0, 0.0, 0.0, 0.0]

        saturnlike_values = [0.0, 0.7, 0.0, 4.0, 0.18, 10.7,
                  1, 1, np.pi/6.74, 0, 1, 1.53,
                  1.95, 0.0, 0.0, 0.0, 0.0]

        mins = [0.0, 0.0, -0.1, 0.0, 0.0, 1.0,
                0.0, 0.9, 0.0, 0.0, 0.0, 1.0,
                1.1, -0.1, -0.1, 0.0, 0.0]

        maxes = [1.0, 1.0, 0.1, 100.0, 1.0, 100.0,
                 1.0, 1.1, np.pi, np.pi, 1.0, 3.0,
                 3.0, 0.1, 0.1, 0.0, 0.0]

        vary_flags = [True, True, False, False, True, True,
                      True, False, True, True, False, True,
                      True, False, False, False, False]


        params = set_params_lm(names, values, mins, maxes, vary_flags)
        params_df = pd.DataFrame(list(zip(values, saturnlike_values, mins, maxes)), columns=['values', 'saturnlike_values', 'mins', 'maxes'], index=names)
        vary_dic = dict(zip(names, vary_flags))
        params_df = params_df.join(pd.DataFrame.from_dict(vary_dic, orient='index', columns=['vary_flags']))
        df_for_mcmc = params_df[params_df['vary_flags']==True]

        ###土星likeな惑星のパラメータで作成したモデル###
        '''
        saturnlike_params = set_params_lm(names, saturnlike_values, mins, maxes, vary_flags)
        #pdic_saturnlike = make_dic(names, saturnlike_values)
        pdic_saturnlike = params_df['saturnlike_values'].to_dict()
        ymodel = ring_model(t, pdic_saturnlike)
        '''
        pdic = params_df['values'].to_dict()
        ring_res = lmfit.minimize(ring_residual_transitfit, params, args=(t, flux_data, flux_err_data.mean(), names), max_nfev=1000)
        best_ring_res_dict[ring_res.chisqr] = ring_res

    ring_res = sorted(best_ring_res_dict.items())[0][1]
    fig = plt.figure()
    ax_lc = fig.add_subplot(2,1,1) #for plotting transit model and data
    ax_re = fig.add_subplot(2,1,2) #for plotting residuals
    #elapsed_time = time.time() - start
    #print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    ring_flux_model = ring_model_transitfit_from_lmparams(ring_res.params, t)
    noring_flux_model = no_ring_model_transitfit_from_lmparams(no_ring_res.params, t, noringnames)
    binned_lc.errorbar(ax=ax_lc)
    ax_lc.plot(t, ring_flux_model, label='Model w/ ring', color='blue')
    ax_lc.plot(t, noring_flux_model, label='Model w/o ring', color='red')
    residuals_ring = binned_lc - ring_flux_model
    residuals_no_ring = binned_lc - noring_flux_model
    chisq_ring = np.sum((residuals_ring.flux.value/flux_err_data)**2)
    chisq_noring = np.sum((residuals_no_ring.flux.value/flux_err_data)**2)
    residuals_ring.plot(ax=ax_re, color='blue', alpha=0.3,  marker='.', zorder=1)
    residuals_no_ring.plot(ax=ax_re, color='red', alpha=0.3,  marker='.', zorder=1)
    ax_re.plot(t, np.zeros(len(t)), color='black', zorder=2)
    ax_lc.legend()
    ax_lc.set_title(f'w/ chisq:{chisq_ring:0f} w/o chisq:{chisq_noring:0f} dof:{len(binned_lc)}')
    plt.tight_layout()
    plt.savefig(f'./lmfit_result/{TOInumber}.png', header=False, index=False)
    #plt.show()
    plt.close()
    ring_res_pdict = ring_res.params.valuesdict()

###csvに書き出し###
input_df = pd.DataFrame.from_dict(params.valuesdict(), orient="index",columns=["input_value"])
output_df = pd.DataFrame.from_dict(ring_res.params.valuesdict(), orient="index",columns=["output_value"])
input_df=input_df.applymap(lambda x: '{:.6f}'.format(x))
output_df=output_df.applymap(lambda x: '{:.6f}'.format(x))
df = input_df.join((output_df, pd.Series(vary_flags, index=names, name='vary_flags')))
#df.to_csv('/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/fitting_result/data/fitting_result_{}_{:.0f}.csv'.format(datetime.datetime.now().strftime('%y%m%d%H%M'), chi_square), header=True, index=False)
df.to_csv(f'./fitting_result/data/{TOInumber}.csv', header=True, index=False)
#fit_report = lmfit.fit_report(ring_res)
#print(fit_report)
sys.exit()


###mcmc setting###
for try_n in range(5):
    mcmc_df = params_df[params_df['vary_flags']==True]
    mcmc_params = mcmc_df.index.to_list()
    for i, param in enumerate(mcmc_params):
        mcmc_df.iloc[i, 0] = ring_res_pdict[param]
    mcmc_pvalues = mcmc_df['values'].values
    #vary_dic = make_dic(names, vary_flags)
    ###generate initial value for theta, phi
    mcmc_df.at['theta', 'values'] = np.random.uniform(low=mcmc_df.at['theta', 'mins'], high=mcmc_df.at['theta', 'maxes'])
    mcmc_df.at['phi', 'values'] = np.random.uniform(low=mcmc_df.at['phi', 'mins'], high=mcmc_df.at['phi', 'maxes'])
    print('mcmc_params: ', mcmc_params)
    print('mcmc_pvalues: ', mcmc_pvalues)
    pos = mcmc_pvalues + 1e-5 * np.random.randn(32, len(mcmc_pvalues))
    #pos = np.array([rp_rs, theta, phi, r_in, r_out]) + 1e-8 * np.random.randn(32, 5)
    nwalkers, ndim = pos.shape
    #filename = "emcee_{0}.h5".format(datetime.datetime.now().strftime('%y%m%d%H%M'))
    #backend = emcee.backends.HDFBackend(filename)
    #backend.reset(nwalkers, ndim)
    max_n = 10000
    index = 0
    autocorr = np.empty(max_n)
    old_tau = np.inf
    #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(t, flux, error_scale), pool=pool)

    #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(t, flux, error_scale), backend=backend)

    ###mcmc run###
    #sampler.run_mcmc(pos, max_n, progress=True)
    if __name__ ==  '__main__':
        with Pool(processes=4) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(t, flux_data, flux_err_data.mean(), mcmc_params), pool=pool)
            #pos = sampler.run_mcmc(pos, max_n)
            #sampler.reset()
            for sample in sampler.sample(pos, iterations=max_n, progress=True):
                # Only check convergence every 100 steps
                if sampler.iteration % 100:
                    continue

                # Compute the autocorrelation time so far
                # Using tol=0 means that we'll always get an estimate even
                # if it isn't trustworthy
                tau = sampler.get_autocorr_time(tol=0)
                autocorr[index] = np.mean(tau)
                index += 1

                # Check convergence
                converged = np.all(tau * 100 < sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                if converged:
                    break
                old_tau = tau


        ###the autocorrelation time###
        n = 100 * np.arange(1, index + 1)
        y = autocorr[:index]
        plt.plot(n, n / 100.0, "--k")
        plt.plot(n, y)
        plt.xlim(0, n.max())
        plt.ylim(0, y.max() + 0.1 * (y.max() - y.min()))
        plt.xlabel("number of steps")
        plt.ylabel(r"mean $\hat{\tau}$")
        plt.savefig(f'tau_{try_n}.png')\
        ##plt.show()
        plt.close()
        print(tau)


        ###step visualization###
        fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
        samples = sampler.get_chain()
        #labels = ['rp_rs', 'theta', 'phi', 'r_in', 'r_out']
        #labels = ['theta', 'phi']
        labels = mcmc_params
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number");
        plt.savefig(f'step_{try_n}.png')
        ##plt.show()
        plt.close()
        ##plt.show()

        ###corner visualization###
        samples = sampler.flatchain
        flat_samples = sampler.get_chain(discard=2500, thin=15, flat=True)
        print(flat_samples.shape)
        """
        truths = []
        for param in labels:
            truths.append(pdic_saturnlike[param])
        fig = corner.corner(flat_samples, labels=labels, truths=truths);
        """
        fig = corner.corner(flat_samples, labels=labels);
        plt.savefig(f'corner_{try_n}.png')
        ##plt.show()
        plt.close()

        """
        tau = sampler.get_autocorr_time()
        burnin = int(2 * np.max(tau))
        thin = int(0.5 * np.min(tau))
        samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)

        print("burn-in: {0}".format(burnin))
        print("thin: {0}".format(thin))
        print("flat chain shape: {0}".format(samples.shape))
        """


        samples = sampler.flatchain
        flat_samples = sampler.get_chain(discard=1000, thin=15, flat=True)
        print(flat_samples.shape)
        inds = np.random.randint(len(flat_samples), size=100)
        plt.errorbar(t, flux_data, yerr=flux_err_data, fmt=".k", capsize=0, alpha=0.1)
        for ind in inds:
            sample = flat_samples[ind]
            model = ring_model(t, pdic, sample)
            plt.plot(t, model, "C1", alpha=0.5)
        #plt.plot(t, ymodel, "k", label="truth")
        plt.legend(fontsize=14)
        #plt.xlim(0, 10)
        plt.xlabel("orbital phase")
        plt.ylabel("flux");
        plt.savefig(f"mcmc_result_{try_n}.png")
        ##plt.show()
        plt.close()


        #import pdb; pdb.set_trace()
        #flux_model = no_ring_model_transitfit_from_lmparams(out.params, t, noringnames)
        #flux_model = ring_model_transitfit_from_lmparams(out.params, t)
        #plt.errorbar(time, flux_data,flux_err_data, label='data', fmt='.k', linestyle=None)
        #folded_lc.errorbar()
        #plt.plot(t, flux_model, label='fit_model')
        #plt.plot(t, ymodel, label='model')
        #plt.legend()
        #plt.savefig('/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/fitting_result/figure/fitting_result_{}_{:.0f}.png'.format(datetime.datetime.now().strftime('%y%m%d%H%M'), chi_square), header=False, index=False)
        ##plt.show()

        ###csvに書き出し###
        #input_df = pd.DataFrame.from_dict(params.valuesdict(), orient="index",columns=["input_value"])
        input_df = pd.DataFrame.from_dict(params.valuesdict(), orient="index",columns=["input_value"])
        output_df = pd.DataFrame.from_dict(ring_res.params.valuesdict(), orient="index",columns=["output_value"])
        input_df=input_df.applymap(lambda x: '{:.6f}'.format(x))
        output_df=output_df.applymap(lambda x: '{:.6f}'.format(x))
        #df = input_df.join((output_df, pd.Series(vary_flags, index=noringnames, name='vary_flags')))
        df = input_df.join((output_df, pd.Series(vary_flags, index=names, name='vary_flags')))
        #df.to_csv('/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/fitting_result/data/fitting_result_{}_{:.0f}.csv'.format(datetime.datetime.now().strftime('%y%m%d%H%M'), chi_square), header=True, index=False)
        df.to_csv(f'./fitting_result_{TOInumber}_{m}.csv', header=True, index=False)
        #fit_report = lmfit.fit_report(ring_res)
        #print(fit_report)


        #import pdb; pdb.set_trace()
        ###use lightkurve(diffrent method from Aizawa+2018)###
        '''
        kic = "KIC10666592"
        tpf = lk.search_targetpixelfile(kic, author="Kepler", cadence="short").download()
        #tpf.plot(frame=100, scale='log', show_colorbar=True)
        lc = tpf.to_lightcurve(aperture_mask=tpf.pipeline_mask)
        #lc.plot()
        period = np.linspace(1, 3, 10000)
        bls = lc.to_periodogram(method='bls', period=period, frequency_factor=500);

        period=2.20473541 #day
        transit_time=121.3585417
        duration=0.162026
        a_rs=4.602
        b=0.224
        rp_rs=0.075522
        i=87.21 * 0.0175 #radian
        a=0.0376 #Orbit Semi-Major Axis [au]
        a=562487835826.56 #cm
        rstar=1.952 * 6.9634 * 10**10 #Rstar cm
        #lc_list = preprocess_each_lc(lc, duration, period, transit_time)
        #transit_time_per_exposure = (duration * len(lc_list) / (lc.time[-1].value - lc.time[0].value))
        #lc = lc.bin(bins=int(300 // transit_time_per_exposure))
        tot = (rstar/a)* (np.sqrt(np.square(1+rp_rs)-np.square(b)) / np.sin(i))
        Ttot = (period/np.pi) * np.arcsin(tot)
        full = (rstar/a)* (np.sqrt(np.square(1-rp_rs)-np.square(b)) / np.sin(i))
        Tfull = (period/np.pi) * np.arcsin(full)
        ingress = (Ttot-Tfull) / 2 #day
        _ = ingress/period
        lc_list = preprocess_each_lc(lc, duration, period, transit_time)
        folded_lc = folding_each_lc(lc_list)
        folded_lc = folded_lc[folded_lc.time > -0.1]
        folded_lc = folded_lc[folded_lc.time < 0.1]
        import pdb; pdb.set_trace()
        folded_lc.errorbar()
        #folded_lc.write('folded_lc.csv', overwrite=True)
        plt.savefig('folded_lc.png')
        ##plt.show()
        plt.close()
        '''
        """for TESS data
        #使う行のみ抽出
        df=pd.read_csv('/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/toi-catalog.csv', encoding='shift_jis')
        df.columns=df.iloc[3].values
        df=df[4:]
        #カラムを入れ替える。
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
        #SN比 ≧ 100のデータを抽出。
        df2 = df[df['Signal-to-noise'] >= 100]
        df2 = df2.reset_index()
        #df2 = df2.drop(columns='index')
        df2.head()
        import pdb; pdb.set_trace()
        TIClist = df2['TIC'].apply(lambda x:int(x))
        for TIC in TIClist:
            tpf = lk.search_targetpixelfile('TIC {}'.format(TIC), mission='TESS', cadence="short").download()
            #tpf.plot(frame=100, scale='log', show_colorbar=True)
            lc = tpf.to_lightcurve(aperture_mask=tpf.pipeline_mask)
            lc.plot()
            #plt.show()
            import pdb; pdb.set_trace()

        #get transit paramter from TESS database
        period=2.20473541
        transit_time=121.3585417
        duration=0.162026

        lc_list = preprocess_each_lc(lc, duration, period, transit_time)
        folded_lc = folding_each_lc(lc_list)
        folded_lc.errorbar()
        #plt.show()
        #すべてのlightcurveの可視化
        lc_collection = search_result.download_all()
        lc_collection.plot();
        #plt.show()
        import pdb; pdb.set_trace()

        #tpf.plot(frame=100, scale='log', show_colorbar=True)
        lc = tpf.to_lightcurve(aperture_mask=tpf.pipeline_mask)
        #lc.plot()
        period = np.linspace(1, 3, 10000)
        bls = lc.to_periodogram(method='bls', period=period, frequency_factor=500);

        period=2.20473541
        transit_time=121.3585417
        duration=0.162026

        lc_list = preprocess_each_lc(lc, duration, period, transit_time)
        folded_lc = folding_each_lc(lc_list)
        folded_lc.errorbar()
        ##plt.show()
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
        df2[df2['TIC']=='142087638']
        #planet_b_period = float(df2[df2['TIC']=='142087638']['Orbital Period Value'].values[0])
        #planet_b_t0 = float(df2[df2['TIC']=='142087638']['Epoch Value'].values[0])
        #planet_b_dur = float(df2[df2['TIC']=='142087638']['Transit Duration Value'].values[0])
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
        ##plt.show()
        plt.cla()
        plt.clf()
        # read the data file (xdata, ydata, yerr)
        xdata = lc.fold(planet_b_period, planet_b_t0).phase.value
        ydata = lc.fold(planet_b_period, planet_b_t0).flux.value
        yerr  = lc.fold(planet_b_period, planet_b_t0).flux_err.value
        """
        '''
        ###土星likeな惑星のパラメータで作成したlight curve###
        error_scale = 0.0001
        eps_data = np.random.normal(size=t.size, scale=error_scale)
        flux = ymodel + eps_data
        '''
