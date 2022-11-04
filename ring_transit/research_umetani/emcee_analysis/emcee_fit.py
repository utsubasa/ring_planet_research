# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import svd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import lightkurve as lk
import warnings
import c_compile_ring
import emcee
import corner
from multiprocessing import Pool, cpu_count, freeze_support
from astropy.io import ascii
import os
import sys
from concurrent import futures
warnings.filterwarnings('ignore')


def ring_model(t, pdic, mcmc_pvalues=None):
    # Ring model
    # Input "x" (1d array), "pdic" (dic)
    # Ouput flux (1d array)
    if mcmc_pvalues is None:
        pass
    else:
        for i, param in enumerate(mcmc_pvalues):
            #print(i, v[i])
            pdic[param] = mcmc_pvalues[i]

    q1, q2, t0, porb, rp_rs, a_rs, b, norm \
            = pdic['q1'], pdic['q2'], pdic['t0'], pdic['porb'], pdic['rp_rs'], pdic['a_rs'], pdic['b'], pdic['norm']
    #theta, phi, tau, r_in, r_out \
            #= np.arcsin( (pdic['b']/pdic['a_rs']) +0.5), pdic['phi'], pdic['tau'], pdic['r_in'], pdic['r_out']
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

def lnlike(mcmc_pvalues, t, y, yerr, pdic):
    return -0.5 * np.sum(((y-ring_model(t, pdic, mcmc_pvalues))/yerr) ** 2)

def log_prior(mcmc_pvalues, mcmc_pnames, df_for_mcmc):
    #p_dict = dict(zip(mcmc_pnames, p))
    #rp_rs, theta, phi, r_in = p
    for i, param in enumerate(mcmc_pnames):
        if df_for_mcmc['mins'][param] <= mcmc_pvalues[i] <= df_for_mcmc['maxes'][param]:
            pass
        else:
            return -np.inf
    #if 0.0 < theta < np.pi/2 and 0.0 < phi < np.pi/2 and 0.0 < rp_rs < 1 and 1.0 < r_in < 7.0:
    #    return 0.0
    return 0.0

def lnprob(mcmc_pvalues, t, y, yerr, mcmc_pnames, df_for_mcmc, pdic):
    lp = log_prior(mcmc_pvalues, mcmc_pnames, df_for_mcmc)
    if not np.isfinite(lp):
        return -np.inf
    chi_square = np.sum(((y-ring_model(t, pdic, mcmc_pvalues))/yerr)**2)
    #print(chi_square)

    return lp + lnlike(mcmc_pvalues, t, y, yerr, pdic)

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
    ax.set_title(f'chisq={chi_square}, dof={len(flux_data)-len(mcmc_df.index)}')
    plt.axis('scaled')
    ax.set_aspect('equal')
    os.makedirs(f'{savedir}/figure/{TOInumber}/illustration', exist_ok=True)
    plt.savefig(f'{savedir}/figure/{TOInumber}/illustration/{file_name}', bbox_inches="tight")
    plt.close()

# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf

# Following the suggestion from Goodman & Weare (2010)
def autocorr_gw2010(y, c=5.0):
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

def load_walkers(h5name):
    sampler = emcee.backends.HDFBackend(h5name)

    return sampler


if __name__ == '__main__':
    freeze_support()
    """各条件の設定"""
    homedir = os.getcwd()
    args = sys.argv
    #p_csv = str(args[1]) 
    #p_csv = 'TOI495.01_65_198.csv'
    angle = str(args[1]) #edge_on, free_angle, face_on
    TOI = str(args[2])

    '''#edge_onを一気に処理したいときはargs[3]にはこちらを割り当てる
    TOIlist = os.listdir(f'{homedir}/{datadir}/fit_p_data/ring_model')
    #TOI = TOIlist[args[3]] 
    '''

    p_csv = os.listdir(f'{homedir}/lmfit_result/fit_p_data/ring_model/{TOI}')[-1] #lmfitのfitting parametersの中で最もF_obsが高いときのパラメータを記録しているcsvを使う
    """読み込むパラメータの準備"""
    TOInumber, _, _ = p_csv.split('_')

    """保存先ディレクトリの設定"""
    csvfile = f'{homedir}/binned_lc_data/{TOInumber}.csv'
    savedir = f'{homedir}/mcmc_result/{angle}'
    mcmc_df = pd.read_csv(f'{homedir}/lmfit_result/fit_p_data/ring_model/{TOInumber}/{p_csv}')
    os.makedirs(f'{savedir}/figure/{TOInumber}', exist_ok=True)
    os.makedirs(f'{savedir}/data/{TOInumber}', exist_ok=True)


    """load lightcurve data"""
    binned_table = ascii.read(csvfile)
    binned_lc = lk.LightCurve(data=binned_table)
    t = binned_lc.time.value
    flux_data = binned_lc.flux.value
    flux_err_data = binned_lc.flux_err.value

    """mcmc setting"""
    p_names = ["q1", "q2", "t0", "porb", "rp_rs", "a_rs",
            "b", "norm", "theta", "phi", "tau", "r_in",
            "r_out", "norm2", "norm3", "ecosw", "esinw"]

    mins = [0.0, 0.0, -0.1, 0.0, 0.01, 1.0,
            0.0, 0.9, 0.0, 0.0, 0.0, 1.00,
            1.00, -0.1, -0.1, 0.0, 0.0]

    maxes = [1.0, 1.0, 0.1, 100.0, 0.5, 100.0,
            1.0, 1.1, np.pi, np.pi, 1.0, 2.45,
            2.45, 0.1, 0.1, 0.0, 0.0]

    vary_flags = [True, True, True, False, True, True,
                True, False, True, True, False, True,
                True, False, False, False, False]
    params_df = pd.DataFrame(list(zip(mins, maxes)), columns=['mins', 'maxes'], index=p_names)
    params_df['vary_flags'] = vary_flags
    df_for_mcmc = params_df[params_df['vary_flags']==True]
    mcmc_df.index = p_names

    """変化させないパラメタの中で値を指定したいものはここでinput_valueとして指定する"""
    if angle == 'edge_on':
        mcmc_df.at['phi', 'input_value'] = 0
        mcmc_df.at['theta', 'input_value'] = np.arcsin(mcmc_df.at['b', 'output_value']/mcmc_df.at['a_rs', 'output_value'])
        phi = 0
        theta = np.arcsin(mcmc_df.at['b', 'output_value']/mcmc_df.at['a_rs', 'output_value'])
    pdic = mcmc_df['input_value'].to_dict()

    #mcmcで事後分布推定しないパラメータをここで設定
    mcmc_df.at['t0', 'vary_flags'] = False
    if angle == 'edge_on':
        mcmc_df.at['phi', 'vary_flags'] = False
        mcmc_df.at['theta', 'vary_flags'] = False
    mcmc_df = mcmc_df[mcmc_df['vary_flags']==True]

    for try_n in range(3):
        """generate initial value for theta, phi"""
        #mcmc_df.at['theta', 'output_value'] = np.arcsin(mcmc_df.at['b', 'output_value']/mcmc_df.at['a_rs', 'output_value'])#+2.22
        if angle == 'free_angle':
            mcmc_df.at['theta', 'output_value'] = np.random.uniform(0.0,np.pi)
            mcmc_df.at['phi', 'output_value'] = np.random.uniform(0.0,np.pi)
        mcmc_df.at['r_in', 'output_value'] = np.random.uniform(1.0,2.45)
        mcmc_df.at['r_out', 'output_value'] = np.random.uniform(1.0,2.45)
        mcmc_pnames = mcmc_df.index.tolist()
        mcmc_init_val = mcmc_df['output_value'].values
        print('mcmc_pnames: ', mcmc_pnames)
        print('mcmc_init_val: ', mcmc_init_val)
        #df_for_mcmc.at['theta', 'mins'] = np.arcsin(mcmc_df.at['b', 'output_value']/mcmc_df.at['a_rs', 'output_value'])+2.21
        #df_for_mcmc.at['theta', 'maxes'] = np.arcsin(mcmc_df.at['b', 'output_value']/mcmc_df.at['a_rs', 'output_value'])+2.23

        """mcmc run"""
        pos = mcmc_init_val + 1e-5 * np.random.randn(32, len(mcmc_init_val))
        nwalkers, ndim = pos.shape
        filename = f"{savedir}/data/{TOInumber}/emcee_{try_n}.h5"
        backend = emcee.backends.HDFBackend(filename)
        #print(f"Initial size: {backend.iteration}")
        #sys.exit()
        backend.reset(nwalkers, ndim)

        max_n = 30000
        discard = 5000
        index = 0

        with Pool(processes=cpu_count()) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(t, flux_data, flux_err_data.mean(), mcmc_pnames, df_for_mcmc, pdic), pool=pool, backend=backend)
            sampler.run_mcmc(pos, max_n, progress=True, store=True)
            #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(t, flux_data, flux_err_data.mean(), mcmc_pnames, df_for_mcmc, pdic), backend=backend)
            #sampler.run_mcmc(pos, max_n, progress=False, store=True)

        """保存したwalkerをロードする場合はこの#を外す"""
        #sampler = load_walkers('/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/fitting_result/data/mcmc_result/free_angle/data/TOI183.01/emcee_0.h5')

        """step visualization"""
        fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
        samples = sampler.get_chain()
        labels = mcmc_pnames
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number");

        plt.savefig(f'{savedir}/figure/{TOInumber}/step_{try_n}.pdf')
        plt.close()
        ##plt.show()



        """corner visualization"""
        flat_samples = sampler.get_chain(discard=discard, thin=15, flat=True)
        print(flat_samples.shape)
        '''
        truths = []
        for param in labels:
            truths.append(pdic_saturnlike[param])
        fig = corner.corner(flat_samples, labels=labels, truths=truths);
        '''
        fig = corner.corner(flat_samples, labels=labels);
        plt.savefig(f'{savedir}/figure/{TOInumber}/corner_{try_n}.pdf')
        ##plt.show()
        plt.close()

        """ライトカーブとモデルのプロット"""
        inds = np.random.randint(len(flat_samples), size=100)
        plt.errorbar(t, flux_data, yerr=flux_err_data, fmt=".k", capsize=0, alpha=0.1)
        for ind in inds:
            sample = flat_samples[ind]
            flux_model = ring_model(t, pdic, sample)
            plt.plot(t, flux_model, "C1", alpha=0.5)
            #fit_report = lmfit.fit_report(ring_res)
            #print(fit_report)
        #plt.plot(t, ymodel, "k", label="truth")
        #plt.xlim(0, 10)
        plt.xlabel("orbital phase")
        plt.ylabel("flux")
        plt.title(f'n_bins: {len(binned_lc)}')
        plt.savefig(f"{savedir}/figure/{TOInumber}/fit_result_{try_n}.pdf")
        ##plt.show()
        plt.close()

        """ポンチ絵の作成とcsvへの保存"""
        for ind in inds:
            sample = flat_samples[ind]
            flux_model = ring_model(t, pdic, sample)
            
            mcmc_res_df = mcmc_df
            mcmc_res_df['output_value'] = sample
            rp_rs = mcmc_res_df.at['rp_rs', 'output_value']
            rin_rp = mcmc_res_df.at['r_in', 'output_value']
            rout_rin = mcmc_res_df.at['r_out', 'output_value']
            b = mcmc_res_df.at['b', 'output_value']
            theta = mcmc_res_df.at['theta', 'output_value']
            phi = mcmc_res_df.at['phi', 'output_value']
            """リングの角度固定の場合はこっち"""
            #theta = theta
            #phi = phi

            chi_square = np.sum(((flux_model-flux_data)/flux_err_data)**2)
            file_name = f'{TOInumber}_{chi_square:.0f}_{try_n}.pdf'
            plot_ring(rp_rs, rin_rp, rout_rin, b, theta, phi, file_name)
            """csvに書き出し"""
            os.makedirs(f'{savedir}/data/{TOInumber}', exist_ok=True)
            mcmc_res_df.to_csv(f'{savedir}/data/{TOInumber}/{TOInumber}_{chi_square:.0f}_{try_n}.csv', header=True, index=False)

        """the autocorrelation time"""
        # Compute the estimators for a few different chain lengths
        chain = sampler.get_chain()[:, :, 0].T
        N = np.exp(np.linspace(np.log(100), np.log(chain.shape[1]), 10)).astype(int)
        gw2010 = np.empty(len(N))
        new = np.empty(len(N))
        for i, n in enumerate(N):
            gw2010[i] = autocorr_gw2010(chain[:, :n])
            new[i] = autocorr_new(chain[:, :n])

        # Plot the comparisons
        plt.loglog(N, gw2010, "o-", label="G&W 2010")
        plt.loglog(N, new, "o-", label="new")
        ylim = plt.gca().get_ylim()
        plt.plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")
        plt.ylim(ylim)
        plt.xlabel("number of samples, $N$")
        plt.ylabel(r"$\tau$ estimates")
        plt.legend(fontsize=14);
        plt.savefig(f'{savedir}/figure/{TOInumber}/tau_{try_n}.pdf')
        #plt.show()
        plt.close()

            

        '''fitting_data/data上で実行し、最もカイ2乗の小さいパラメータcsvを取得
        TOIs = os.listdir()
        list = []
        for TOI in TOIs:
            os.chdir(TOI)
            files = os.listdir()
            file = sorted(files, key=natural_keys)[0]
            list.append(file)
        os.chdir('..')
    '''