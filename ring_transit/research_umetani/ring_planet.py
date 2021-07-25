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
from astropy.table import Table

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
        ##import pdb; pdb.set_trace()
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
    global chi_square
    model = ring_model(x, params.valuesdict())
    chi_square = np.sum(((data-model)/eps_data)**2)
    #print(params)
    print(((data-model)/eps_data)**2)
    print(chi_square)
    print(np.max(((data-model)/eps_data)**2))

    return (data-model) / eps_data

#リングありモデルをfitting
def no_ring_residual_transitfit(params, x, data, eps_data, names):
    global chi_square
    params_batman = set_params_batman(params, names)
    m = batman.TransitModel(params_batman, x)    #initializes model
    model = m.light_curve(params_batman)         #calculates light curve
    chi_square = np.sum(((data-model)/eps_data)**2)
    print(params)
    print(chi_square)
    return (data-model) / eps_data

def ring_model_transitfit_from_lmparams(params, x):
    model = ring_model(x, params.valuesdict())         #calculates light curve
    return model

def no_ring_model_transitfit_from_lmparams(params, x, names):
    params_batman = set_params_batman(params, names)
    m = batman.TransitModel(params_batman, x)    #initializes model
    model = m.light_curve(params_batman)
    return model

def lnlike(v, t, y, yerr):
    return -0.5 * np.sum(((y-ring_model(t, pdic_saturnlike, v))/yerr) ** 2)

def log_prior(v):
    #p_dict = dict(zip(mcmc_params, p))
    #rp_rs, theta, phi, r_in = p
    for i, param in enumerate(mcmc_params):
        if df_for_mcmc['mins'][param] <= v[i] <= df_for_mcmc['maxes'][param]:
            pass
        else:
            return -np.inf
    #if 0.0 < theta < np.pi/2 and 0.0 < phi < np.pi/2 and 0.0 < rp_rs < 1 and 1.0 < r_in < 7.0:
    #    return 0.0
    return 0.0


def lnprob(v, x, y, yerr):
    lp = log_prior(v)
    if not np.isfinite(lp):
        return -np.inf
    chi_square = np.sum(((y-ring_model(x, pdic_saturnlike, v))/yerr)**2)
    print(chi_square)
    return lp + lnlike(v, x, y, yerr)

"""use lightkurve(diffrent method from Aizawa+2018)"""
kic = "KIC10666592"
tpf = lk.search_targetpixelfile(kic, author="Kepler", cadence="short").download()
#tpf.plot(frame=100, scale='log', show_colorbar=True)
lc = tpf.to_lightcurve(aperture_mask=tpf.pipeline_mask)
lc.plot()
period = np.linspace(1, 3, 10000)
bls = lc.to_periodogram(method='bls', period=period, frequency_factor=500);

period=2.20473541
transit_time=121.3585417
duration=0.162026

planet_b_model = bls.get_transit_model(period=period,transit_time=transit_time,duration=duration)
ax = lc.scatter()
#planet_b_model.plot(ax=ax, c='dodgerblue', label='Planet b Transit Model');
minId = signal.argrelmin(lc.flux.value, order=3000)
plt.plot(lc.time.value[minId], lc.flux.value[minId], "bo")
plt.show()
half_duration = (duration/2)*24*60
twice_duration = (duration*2)*24*60 #durationを2倍、単位をday→mi
lc_cut_point = half_duration + twice_duration

"""params setting"""
noringnames = ["t0", "per", "rp", "a", "inc", "ecc", "w", "q1", "q2"]
#values = [0.0, 4.0, 0.08, 8.0, 83.0, 0.0, 90.0, 0.2, 0.2]
values = [transit_time, period, 0.08, 8.0, 83.0, 0.0, 90.0, 0.2, 0.2]
mins = [-0.1, 4.0, 0.03, 4, 80, 0, 90, 0.0, 0.0]
maxes = [0.1, 4.0, 0.2, 20, 110, 0, 90, 1.0, 1.0]
#vary_flags = [True, False, True, True, True, False, False, True, True]
vary_flags = [False, False, True, True, True, False, False, True, True]
params = set_params_lm(noringnames, values, mins, maxes, vary_flags)

for transitId in minId[0]:
    """transit fitting and clip outliers"""
    start = int(transitId - lc_cut_point)
    end = int(transitId + lc_cut_point)
    target_lc = lc[start:end].normalize()
    out = lmfit.minimize(no_ring_residual_transitfit,params,args=(target_lc.time.value, target_lc.flux.value, target_lc.flux_err.value, noringnames),max_nfev=1000)
    target_lc.errorbar()
    flux_model = no_ring_model_transitfit_from_lmparams(out.params, target_lc.time.value, noringnames)
    lk.LightCurve()
    plt.plot(target_lc.time.value, flux_model, label='fit_model')
    #plt.plot(t, ymodel, label='model')
    plt.legend()
    #plt.savefig('/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/fitting_result/figure/fitting_result_{}_{:.0f}.png'.format(datetime.datetime.now().strftime('%y%m%d%H%M'), chi_square), header=False, index=False)
    plt.show()
    import pdb; pdb.set_trace()

    """curve fiting"""
    target_lc = target_lc.to_pandas()
    before_transit = target_lc[target_lc.index < transit_time-duration/2]
    after_transit = target_lc[target_lc.index > transit_time+duration/2]
    out_transit = pd.concat([before_transit, after_transit])
    out_transit = out_transit.reset_index()
    out_transit = Table.from_pandas(out_transit)
    out_transit = lk.LightCurve(data=out_transit)
    model = lmfit.models.PolynomialModel()
    params = model.make_params(c0=1, c1=1, c2=1, c3=1, c4=1, c5=1, c6=1, c7=1)
    result = model.fit(out_transit.flux.value, params, x=out_transit.time.value)
    result.plot()



    import pdb; pdb.set_trace()





    #model fitting

lc = lc.normalize()
import pdb; pdb.set_trace()
flat, trend = lc.flatten(window_length=301, return_trend=True)
ax = lc.errorbar(label="Kepler-2")
trend.plot(ax=ax, color='red', lw=2, label='Trend')
flat.errorbar(label="Kepler-2")
folded_lc = flat.fold(period=2.20473541, epoch_time=122.763305)
folded_lc.errorbar()
plt.show()
##import pdb; pdb.set_trace()
time = folded_lc.time.value
flux_data = folded_lc.flux.value
flux_err_data = folded_lc.flux_err.value



#lm.minimizeのためのparamsのセッティング。これはリングありモデル
"""parameters setting"""

'''
names = ["q1", "q2", "t0", "porb", "rp_rs", "a_rs",
         "b", "norm", "theta", "phi", "tau", "r_in",
         "r_out", "norm2", "norm3", "ecosw", "esinw"]
#values = [0.2, 0.2, 0.0, 4.0, (float(df2[df2['TIC']=='142087638']['Planet Radius Value'].values[0])*0.0091577) / float(df2[df2['TIC']=='142087638']['Star Radius Value'].values[0]), 40.0,
#          0.5, 1.0, 45.0, 45.0, 0.5, 1.5,
#          2.0/1.5, 0.0, 0.0, 0.0, 0.0]
values = [0.0, 0.7, 0.0, 4.0, 0.5, 10.7,
          1, 1, np.pi/6.0, np.pi/9.0, 1, 1.13,
          2.95, 0.0, 0.0, 0.0, 0.0]

saturnlike_values = [0.0, 0.7, 0.0, 4.0, 0.18, 10.7,
          1, 1, np.pi/6.74, 0, 1, 1.53,
          1.95, 0.0, 0.0, 0.0, 0.0]

mins = [0.0, 0.0, -0.0001, 0.0, 0.0, 1.0,
        0.0, 0.9, 0.0, 0.0, 0.0, 1.0,
        1.1, -0.1, -0.1, 0.0, 0.0]

maxes = [1.0, 1.0, 0.0001, 100.0, 1.0, 1000.0,
         1.0, 1.1, np.pi/2, np.pi/2, 1.0, 7.0,
         10.0, 0.1, 0.1, 0.0, 0.0]

vary_flags = [False, False, False, False, True, False,
              False, False, True, True, False, True,
              True, False, False, False, False]
params = set_params_lm(names, values, mins, maxes, vary_flags)
params_df = pd.DataFrame(list(zip(values, saturnlike_values, mins, maxes)), columns=['values', 'saturnlike_values', 'mins', 'maxes'], index=names)
vary_dic = dict(zip(names, vary_flags))
params_df = params_df.join(pd.DataFrame.from_dict(vary_dic, orient='index', columns=['vary_flags']))
df_for_mcmc = params_df[params_df['vary_flags']==True]
t = np.linspace(-0.2, 0.2, 300)

"""土星likeな惑星のパラメータで作成したモデル"""
saturnlike_params = set_params_lm(names, saturnlike_values, mins, maxes, vary_flags)
#pdic_saturnlike = make_dic(names, saturnlike_values)
pdic_saturnlike = params_df['saturnlike_values'].to_dict()
#pdic = make_dic(names, values)
pdic = params_df['values'].to_dict()
ymodel = ring_model(t, pdic_saturnlike)

"""土星likeな惑星のパラメータで作成したlight curve"""
error_scale = 0.0001
eps_data = np.random.normal(size=t.size, scale=error_scale)
flux = ymodel + eps_data





"""ring model fitting by minimizing chi_square"""
#out = lmfit.minimize(ring_residual_transitfit, params, args=(t, flux, error_scale, names), max_nfev=1000)
#out = lmfit.minimize(ring_residual_transitfit, params, args=(time, flux_data, flux_err_data, names), max_nfev=10000)
out = lmfit.minimize(ring_residual_transitfit, params, args=(time, flux_data, flux_err_data, names), max_nfev=10000)
out_pdict = out.params.valuesdict()
#import pdb; pdb.set_trace()

"""mcmc setting"""
mcmc_df = params_df[params_df['vary_flags']==True]
mcmc_params = mcmc_df.index.to_list()
for i, param in enumerate(mcmc_params):
    mcmc_df.iloc[i, 0] = out_pdict[param]
mcmc_pvalues = mcmc_df['values'].values
#vary_dic = make_dic(names, vary_flags)
print('mcmc_params: ', mcmc_params)
print('mcmc_pvalues: ', mcmc_pvalues)
pos = mcmc_pvalues + 1e-5 * np.random.randn(32, len(mcmc_pvalues))
#pos = np.array([rp_rs, theta, phi, r_in, r_out]) + 1e-8 * np.random.randn(32, 5)
nwalkers, ndim = pos.shape


#filename = "emcee_{0}.h5".format(datetime.datetime.now().strftime('%y%m%d%H%M'))
#backend = emcee.backends.HDFBackend(filename)
#backend.reset(nwalkers, ndim)


max_n = 11000
index = 0
autocorr = np.empty(max_n)
old_tau = np.inf
'''
if __name__ ==  '__main__':
    '''
    with Pool() as pool:
        #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(t, flux, error_scale), pool=pool)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(t, flux, error_scale), backend=backend)
        start = time.time()
        """mcmc run"""
        #sampler.run_mcmc(pos, max_n, progress=True)
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
        end = time.time()
        multi_time = end - start
        print("Multiprocessing took {0:.1f} seconds".format(multi_time))

    """the autocorrelation time"""
    n = 100 * np.arange(1, index + 1)
    y = autocorr[:index]
    plt.plot(n, n / 100.0, "--k")
    plt.plot(n, y)
    plt.xlim(0, n.max())
    plt.ylim(0, y.max() + 0.1 * (y.max() - y.min()))
    plt.xlabel("number of steps")
    plt.ylabel(r"mean $\hat{\tau}$")
    plt.show()

    """step visualization"""
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
    plt.show()

    """corner visualization"""
    samples = sampler.flatchain
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    print(flat_samples.shape)
    truths = []
    for param in mcmc_params:
        truths.append(pdic_saturnlike[param])
    fig = corner.corner(samples, labels=labels, truths=truths);
    plt.show()

    tau = sampler.get_autocorr_time()
    burnin = int(2 * np.max(tau))
    thin = int(0.5 * np.min(tau))
    samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)

    print("burn-in: {0}".format(burnin))
    print("thin: {0}".format(thin))
    print("flat chain shape: {0}".format(samples.shape))

    """
    samples = sampler.flatchain
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    print(flat_samples.shape)
    """
    """
    inds = np.random.randint(len(flat_samples), size=100)
    for ind in inds:
        sample = flat_samples[ind]
        plt.plot(t, np.dot(np.vander(t, 2), sample[:2]), "C1", alpha=0.1)
    """
    for s in samples[np.random.randint(len(samples), size=24)]:
        plt.plot(t, ring_model(t, pdic, v), color="#4682b4", alpha=0.3)
    plt.errorbar(t, flux, yerr=error_scale, fmt=".k", capsize=0)
    plt.plot(t, ymodel, "k", label="truth")
    plt.legend(fontsize=14)
    #plt.xlim(0, 10)
    plt.xlabel("t")
    plt.ylabel("flux");
    plt.show()
    '''

    #import pdb; pdb.set_trace()
    #flux_model = no_ring_model_transitfit_from_lmparams(out.params, t, noringnames)
    flux_model = ring_model_transitfit_from_lmparams(out.params, time)
    #plt.errorbar(time, flux_data,flux_err_data, label='data', fmt='.k', linestyle=None)
    folded_lc.errorbar()
    plt.plot(time, flux_model, label='fit_model')
    #plt.plot(t, ymodel, label='model')
    plt.legend()
    #plt.savefig('/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/fitting_result/figure/fitting_result_{}_{:.0f}.png'.format(datetime.datetime.now().strftime('%y%m%d%H%M'), chi_square), header=False, index=False)
    plt.show()

    """csvに書き出し"""
    #input_df = pd.DataFrame.from_dict(params.valuesdict(), orient="index",columns=["input_value"])
    input_df = pd.DataFrame.from_dict(saturnlike_params.valuesdict(), orient="index",columns=["input_value"])
    output_df = pd.DataFrame.from_dict(out.params.valuesdict(), orient="index",columns=["output_value"])
    input_df=input_df.applymap(lambda x: '{:.6f}'.format(x))
    output_df=output_df.applymap(lambda x: '{:.6f}'.format(x))
    #df = input_df.join((output_df, pd.Series(vary_flags, index=noringnames, name='vary_flags')))
    df = input_df.join((output_df, pd.Series(vary_flags, index=names, name='vary_flags')))
    df.to_csv('/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/fitting_result/data/fitting_result_{}_{:.0f}.csv'.format(datetime.datetime.now().strftime('%y%m%d%H%M'), chi_square), header=True, index=False)
    fit_report = lmfit.fit_report(out)


    import pdb; pdb.set_trace()


    """
    parfile  = "/Users/u_tsubasa/work/ring_planet_research/ring_transit/python_ext/exoring_test/test/para_result_ring.dat"
    parvalues = []
    file = open(parfile, "r")
    lines = file.readlines()
    for (i, line) in enumerate(lines):

        itemList = line.split()
        parvalues.append(float(itemList[1]))


    parvalues = np.array(parvalues)
    """

    """
    names = ["t0", "per", "rp", "a", "inc", "ecc", "w", "q1", "q2"]
    values = [0, 3.5, 0.08, 8, 83, 0, 90, 0.2, 0.2]
    mins = [0, 3.5, 0.03, 4, 80, 0, 90, 0.0, 0.0]
    maxes = [0, 3.5, 0.2, 20, 110, 0, 90, 1.0, 1.0]
    vary_flags = [False, False, True, True, True, False, False, True, True]
    pdic = make_dic(names, values)
    #import pdb; pdb.set_trace()
    out = lmfit.minimize(ring_residual_transitfit, params, args=(xdata, ydata, yerr, names))
    flux_model = model_transitfit_from_lmparams(out.params, xdata, names)
    print(lmfit.fit_report(out.params))


    #print test data
    #pdic = make_dic(names, parvalues)
    #plt.plot(xdata, flux_model, label='model')
    #plt.plot(xdata, ydata, label='data')
    """

'''for TESS data
"""使う行のみ抽出"""
df=pd.read_csv('/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/toi-catalog.csv', encoding='shift_jis')
df.columns=df.iloc[3].values
df=df[4:]
"""カラムを入れ替える。"""
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
"""SN比 ≧ 100のデータを抽出。"""
df2 = df[df['Signal-to-noise'] >= 100]
df2 = df2.reset_index()
#df2 = df2.drop(columns='index')
df2.head()
search_result = lk.search_lightcurve('TIC 142087638', mission='TESS', exptime=120)
"""すべてのlightcurveの可視化"""
lc_collection = search_result.download_all()
#lc_collection.plot();
"""### foldingを任せた場合
すべての観測を平坦化しノーマライズする。これは"a stitched light curve"で表される。詳しくは Kepler data with Lightkurve.
"""
#lc = lc_collection.stitch().flatten(window_length=901).remove_outliers()
#lc.plot();
#lc = lc_collection.stitch().flatten(window_length=901).remove_outliers()
"""### foldingを自分でやる場合"""
lc_single = search_result[3].download()
#lc_single.plot();
lc = lc_single.flatten(window_length=901).remove_outliers()
#lc.plot();
"""### orbital periodをBLSで探す"""
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
"""### カタログのorbital periodを使う"""
df2[df2['TIC']=='142087638']
#planet_b_period = float(df2[df2['TIC']=='142087638']['Orbital Period Value'].values[0])
#planet_b_t0 = float(df2[df2['TIC']=='142087638']['Epoch Value'].values[0])
#planet_b_dur = float(df2[df2['TIC']=='142087638']['Transit Duration Value'].values[0])
# Check the value for period
print('planet_b_period: ', planet_b_period)
print('planet_b_t0: ', planet_b_t0)
print('planet_b_dur: ', planet_b_dur)
"""### folding"""
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
#plt.show()
plt.cla()
plt.clf()
# read the data file (xdata, ydata, yerr)
xdata = lc.fold(planet_b_period, planet_b_t0).phase.value
ydata = lc.fold(planet_b_period, planet_b_t0).flux.value
yerr  = lc.fold(planet_b_period, planet_b_t0).flux_err.value
'''
