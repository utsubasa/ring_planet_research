# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import svd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
#import lmfit
import lightkurve as lk
from fit_model import model
import warnings
import c_compile_ring
import batman
import datetime
import time
import emcee
import corner
from multiprocessing import Pool
#from lightkurve import search_targetpixelfile
from scipy import signal
from astropy.io import ascii
import glob
import os
import sys
warnings.filterwarnings('ignore')


def ring_model(t, pdic, mcmc_pvalues=None):
    # Ring model
    # Input "x" (1d array), "pdic" (dic)
    # Ouput flux (1d array)
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

def plot_ring(rp_rs, rin_rp, rout_rin, b, theta, phi, file_name):
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
    ax.set_title(f'chisq={str(ring_res.chisqr)[:6]}, dof={ring_res.nfree}')
    plt.axis('scaled')
    ax.set_aspect('equal')
    #os.makedirs(f'./lmfit_result/illustration/{TOInumber}', exist_ok=True)
    os.makedirs(f'./mcmc_result/figure/{TOInumber}/illustration', exist_ok=True)
    #plt.savefig(f'./lmfit_result/illustration/{TOInumber}/{file_name}', bbox_inches="tight")
    plt.savefig(f'./mcmc_result/figure/{TOInumber}/illustration/{file_name}', bbox_inches="tight")

p_names = ["q1", "q2", "t0", "porb", "rp_rs", "a_rs",
         "b", "norm", "theta", "phi", "tau", "r_in",
         "r_out", "norm2", "norm3", "ecosw", "esinw"]

mins = [0.0, 0.0, -0.1, 0.0, 0.003, 1.0,
        0.0, 0.9, 0.0, 0.0, 0.0, 1.0,
        1.1, -0.1, -0.1, 0.0, 0.0]

maxes = [1.0, 1.0, 0.1, 100.0, 1.0, 100.0,
         1.0, 1.1, np.pi, np.pi, 1.0, 3.0,
         3.0, 0.1, 0.1, 0.0, 0.0]

vary_flags = [True, True, False, False, True, True,
              True, False, True, True, False, True,
              True, False, False, False, False]


params_df = pd.DataFrame(list(zip(mins, maxes)), columns=['mins', 'maxes'], index=p_names)
params_df['vary_flags'] = vary_flags
df_for_mcmc = params_df[params_df['vary_flags']==True]

p_csvlist = ['TOI267.01_735_1.csv','TOI585.01_352_14.png','TOI615.01_444_2.png','TOI624.01_847_16.png',
            'TOI665.01_594_0.csv','TOI857.01_462_14.png','TOI1025.01_426_12.png','TOI1092.01_434_19.png',
            'TOI1283.01_450_7.png','TOI1292.01_204_16.png','TOI1431.01_284_19.png','TOI1924.01_548_11.png',
            'TOI1976.01_798_5.png','TOI2020.01_445_6.png','TOI2140.01_232_9.png','TOI3460.01_715_7.png',
            'TOI4606.01_753_12.png']
df = pd.read_csv('./exofop_tess_tois.csv')
df = df[df['Planet SNR']>100]
df['TOI'] = df['TOI'].astype(str)


#ここはファイル名を要素にしたリストでfor loop
for p_csv in p_csvlist:
    #dataの呼び出し
    TOInumber, _, _ = p_csv.split('_')
    param_df = df[df['TOI'] == TOInumber[3:]]
    duration = param_df['Duration (hours)'].values / 24
    csvfile = f'./folded_lc_data/{TOInumber}.csv'
    try:
        folded_table = ascii.read(csvfile)
    except FileNotFoundError:
        continue
    folded_lc = lk.LightCurve(data=folded_table)
    folded_lc = folded_lc[(folded_lc.time.value < duration*0.8) & (folded_lc.time.value > -duration*0.8)]
    import astropy.units as u
    #binned_lc = folded_lc.bin(time_bin_size=1*u.minute).remove_nans()
    binned_lc = folded_lc.bin(bins=500).remove_nans()
    t = binned_lc.time.value
    flux_data = binned_lc.flux.value
    flux_err_data = binned_lc.flux_err.value

    ###mcmc setting###
    mcmc_df = pd.read_csv(f'./fitting_result/data/{TOInumber}/{p_csv}')
    #mcmc_df = pd.read_csv(f'./mcmc_result/fit_pdata/{p_csv}')
    mcmc_df.index = p_names
    pdic = mcmc_df['input_value'].to_dict()
    mcmc_df = mcmc_df[mcmc_df['vary_flags']==True]
    mcmc_params = mcmc_df.index.tolist()
    for try_n in range(5):
        mcmc_pvalues = mcmc_df['output_value'].values
        #vary_dic = make_dic(names, vary_flags)
        ###generate initial value for theta, phi
        mcmc_df.at['theta', 'output_value'] = np.random.uniform(1e-5,np.pi-1e-5)
        mcmc_df.at['phi', 'output_value'] = np.random.uniform(0.0,np.pi)
        print('mcmc_params: ', mcmc_params)
        print('mcmc_pvalues: ', mcmc_pvalues)
        pos = mcmc_pvalues + 1e-5 * np.random.randn(32, len(mcmc_pvalues))
        #pos = np.array([rp_rs, theta, phi, r_in, r_out]) + 1e-8 * np.random.randn(32, 5)
        nwalkers, ndim = pos.shape
        #filename = "emcee_{0}.h5".format(datetime.datetime.now().strftime('%y%m%d%H%M'))
        #backend = emcee.backends.HDFBackend(filename)
        #backend.reset(nwalkers, ndim)
        max_n = 10000
        discard = 2500
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

            '''
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
            '''

            os.makedirs(f'./mcmc_result/figure/{TOInumber}', exist_ok=True)

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
            plt.savefig(f'./mcmc_result/figure/{TOInumber}/step_{try_n}.png')
            ##plt.show()
            plt.close()
            ##plt.show()

            ###corner visualization###
            samples = sampler.flatchain
            flat_samples = sampler.get_chain(discard=discard, thin=15, flat=True)
            print(flat_samples.shape)
            """
            truths = []
            for param in labels:
                truths.append(pdic_saturnlike[param])
            fig = corner.corner(flat_samples, labels=labels, truths=truths);
            """
            fig = corner.corner(flat_samples, labels=labels);
            plt.savefig(f'./mcmc_result/figure/{TOInumber}/corner_{try_n}.png')
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
            inds = np.random.randint(len(flat_samples), size=100)
            plt.errorbar(t, flux_data, yerr=flux_err_data, fmt=".k", capsize=0, alpha=0.1)
            for ind in inds:
                sample = flat_samples[ind]
                model = ring_model(t, pdic, sample)
                plt.plot(t, model, "C1", alpha=0.5)
                ###csvに書き出し###
                mcmc_res_df = mcmc_df
                mcmc_res_df['output_value'] = sample
                #df.to_csv('/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/fitting_result/data/fitting_result_{}_{:.0f}.csv'.format(datetime.datetime.now().strftime('%y%m%d%H%M'), chi_square), header=True, index=False)
                rp_rs = mcmc_res_df.at['rp_rs', 'output_value']
                rin_rp = mcmc_res_df.at['r_in', 'output_value']
                rout_rin = mcmc_res_df.at['r_out', 'output_value']
                b = mcmc_res_df.at['b', 'output_value']
                theta = mcmc_res_df.at['theta', 'output_value']
                phi = mcmc_res_df.at['phi', 'output_value']
                file_name = f'{TOInumber}_{ind}_{try_n}.pdf'
                plot_ring(rp_rs, rin_rp, rout_rin, b, theta, phi, file_name)
                os.makedirs(f'./mcmc_result/fit_pdata/{TOInumber}', exist_ok=True)
                mcmc_res_df.to_csv(f'./mcmc_result/fit_pdata/{TOInumber}/{TOInumber}_{ind}_{try_n}.csv', header=True, index=False)
                #fit_report = lmfit.fit_report(ring_res)
                #print(fit_report)
            #plt.plot(t, ymodel, "k", label="truth")
            plt.legend(fontsize=14)
            #plt.xlim(0, 10)
            plt.xlabel("orbital phase")
            plt.ylabel("flux")
            plt.title(f'n_bins: {len(binned_lc)}')
            plt.savefig(f"./mcmc_result/figure/{TOInumber}/fit_result_{try_n}.png")
            ##plt.show()
            plt.close()
