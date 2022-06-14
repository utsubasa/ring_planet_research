# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import svd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
#import lmfit
import lightkurve as lk
#from fit_model import model
import warnings
import c_compile_ring
#import batman
#import datetime
#import time
import emcee
import corner
from multiprocessing import Pool
from astropy.io import ascii
#import glob
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
            = np.arcsin( (pdic['b']/pdic['a_rs']) +0.5), pdic['phi'], pdic['tau'], pdic['r_in'], pdic['r_out']
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
    ax.set_title(f'chisq={chi_square}, dof={len(flux_data)-len(mcmc_df.index)}')
    plt.axis('scaled')
    ax.set_aspect('equal')
    #os.makedirs(f'./lmfit_result/illustration/{TOInumber}', exist_ok=True)
    os.makedirs(f'./mcmc_result/figure/{TOInumber}/illustration', exist_ok=True)
    #plt.savefig(f'./lmfit_result/illustration/{TOInumber}/{file_name}', bbox_inches="tight")
    plt.savefig(f'./mcmc_result/figure/{TOInumber}/illustration/{file_name}', bbox_inches="tight")
    plt.close()

p_names = ["q1", "q2", "t0", "porb", "rp_rs", "a_rs",
         "b", "norm", "theta", "phi", "tau", "r_in",
         "r_out", "norm2", "norm3", "ecosw", "esinw"]

mins = [0.0, 0.0, -0.1, 0.0, 0.003, 1.0,
        0.0, 0.9, 0, 0.0, 0.0, 1.0,
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

#p_csvlist = ['TOI1963.01_665_6.csv']
#p_csvlist = ['TOI267.01_735_1.csv']


p_csvlist = ['TOI3612.01_139_1.csv', 'TOI453.01_593_5.csv', 'TOI157.01_166_10.csv', 'TOI1796.01_143_17.csv', 'TOI1265.01_376_13.csv', 'TOI423.01_320_0.csv', 'TOI1134.01_876_10.csv', 'TOI2126.01_116_0.csv', 
              'TOI1292.01_185_4.csv', 'TOI1478.01_241_8.csv', 'TOI1937.01_132_6.csv', 'TOI2131.01_371_9.csv', 'TOI924.01_216_0.csv', 'TOI677.01_221_0.csv', 'TOI1825.01_2750_3.csv', 'TOI4059.01_158_0.csv', 
              'TOI527.01_321_3.csv', 'TOI2140.01_232_9.csv', 'TOI1456.01_1226_8.csv', 'TOI2579.01_381_8.csv', 'TOI1615.01_252_5.csv', 'TOI1936.01_282_16.csv', 'TOI199.01_674_17.csv', 'TOI4623.01_226_11.csv', 
              'TOI508.01_398_7.csv', 'TOI818.01_164_8.csv', 'TOI1431.01_284_19.csv', 'TOI2127.01_693_16.csv', 'TOI1135.01_2505_13.csv', 'TOI1264.01_43612_5.csv', 'TOI820.01_352_10.csv', 'TOI1951.01_299_11.csv', 
              'TOI1833.01_488_12.csv', 'TOI1670.01_1786_3.csv', 'TOI822.01_390_17.csv', 'TOI1025.01_426_12.csv', 'TOI1454.01_464_0.csv', 'TOI1826.01_125_7.csv', 'TOI212.01_317_7.csv', 'TOI143.01_362_5.csv', 
              'TOI2020.01_445_6.csv', 'TOI1150.01_2398_17.csv', 'TOI683.01_207_8.csv', 'TOI418.01_145_2.csv', 'TOI766.01_253_0.csv', 'TOI845.01_530_17.csv', 'TOI1271.01_2046_12.csv', 'TOI1934.01_554_14.csv', 
              'TOI1270.01_141_4.csv', 'TOI1312.01_386_0.csv', 'TOI738.01_196_1.csv', 'TOI767.01_301_10.csv', 'TOI1425.01_321_3.csv', 'TOI1455.01_204_2.csv', 'TOI834.01_315_3.csv', 'TOI4138.01_375_12.csv', 
              'TOI1151.01_473_2.csv', 'TOI3846.01_171_1.csv', 'TOI2021.01_254_5.csv', 'TOI675.01_755_9.csv', 'TOI1830.01_1305_14.csv', 'TOI391.01_213_17.csv', 'TOI4516.01_152_12.csv', 'TOI2154.01_196_3.csv', 
              'TOI1248.01_158_4.csv', 'TOI349.01_324_9.csv', 'TOI2025.01_304_18.csv', 'TOI671.01_487_4.csv', 'TOI1823.01_5979_3.csv', 'TOI267.01_735_1.csv', 'TOI1612.01_223_18.csv', 'TOI1283.01_450_7.csv', 
              'TOI1274.01_245_8.csv', 'TOI121.01_282_9.csv', 'TOI1767.01_545_7.csv', 'TOI857.01_462_14.csv', 'TOI1050.01_398_11.csv', 'TOI774.01_120_6.csv', 'TOI4470.01_720_7.csv', 'TOI1682.01_453_0.csv', 
              'TOI1419.01_138_3.csv', 'TOI1909.01_273_6.csv', 'TOI2200.01_261_7.csv', 'TOI201.01_3670_0.csv', 'TOI4486.01_901_11.csv', 'TOI150.01_564_9.csv', 'TOI1845.01_160_4.csv', 'TOI1766.01_499_2.csv', 
              'TOI1295.01_303_16.csv', 'TOI1300.01_365_6.csv', 'TOI934.01_166_0.csv', 'TOI4449.01_135_2.csv', 'TOI1771.01_491_8.csv', 'TOI1420.01_233_2.csv', 'TOI433.01_165_1.csv', 'TOI1019.01_439_6.csv', 
              'TOI4087.01_283_16.csv', 'TOI4162.01_439_0.csv', 'TOI2119.01_257_7.csv', 'TOI899.01_270_8.csv', 'TOI1069.01_198_0.csv', 'TOI2024.01_381_0.csv', 'TOI147.01_196_7.csv', 'TOI264.01_337_2.csv', 
              'TOI135.01_452_19.csv', 'TOI1676.01_89_3.csv', 'TOI1714.01_394_13.csv', 'TOI758.01_739_5.csv', 'TOI1141.01_5997_10.csv', 'TOI665.01_594_5.csv', 'TOI1385.01_3345_3.csv', 'TOI479.01_196_1.csv', 
              'TOI780.01_205_9.csv', 'TOI1302.01_389_9.csv', 'TOI1297.01_307_10.csv', 'TOI615.01_444_2.csv', 'TOI1130.01_280_10.csv', 'TOI781.01_283_2.csv', 'TOI1198.01_1589_14.csv', 'TOI272.01_351_6.csv', 
              'TOI1924.01_548_11.csv', 'TOI123.01_194036_14.csv', 'TOI1259.01_157_0.csv', 'TOI842.01_128_4.csv', 'TOI182.01_189_3.csv', 'TOI224.01_49685_0.csv', 'TOI1810.01_166_2.csv', 'TOI1651.01_286_4.csv', 
              'TOI471.01_259_18.csv', 'TOI1268.01_325_2.csv', 'TOI105.01_380_1.csv', 'TOI959.01_181_3.csv', 'TOI3849.01_211_7.csv', 'TOI625.01_259_4.csv', 'TOI112.01_261_15.csv', 'TOI195.01_264_10.csv', 
              'TOI1186.01_1826_4.csv', 'TOI655.01_129_0.csv', 'TOI194.01_421_0.csv', 'TOI232.01_193_3.csv', 'TOI1647.01_1287_3.csv', 'TOI1725.01_146_1.csv', 'TOI163.01_360_0.csv', 'TOI2000.01_330_9.csv', 
              'TOI769.01_317_1.csv', 'TOI490.01_328_17.csv', 'TOI1012.01_204_12.csv', 'TOI505.01_367_0.csv', 'TOI624.01_847_16.csv', 'TOI1251.01_308_6.csv', 'TOI3492.01_334_7.csv', 'TOI4601.01_334_0.csv', 
              'TOI1148.01_3492_1.csv', 'TOI1861.01_8245_15.csv', 'TOI368.01_278_0.csv', 'TOI1494.01_136_1.csv', 'TOI1811.01_171_4.csv', 'TOI1236.01_282_2.csv', 'TOI2017.01_438_14.csv', 'TOI1573.01_651_1.csv', 
              'TOI2403.01_428_11.csv', 'TOI4545.01_216_2.csv', 'TOI159.01_438_7.csv', 'TOI106.01_401_9.csv', 'TOI3960.01_207_0.csv', 'TOI1165.01_181_5.csv', 'TOI472.01_112_9.csv', 'TOI4535.01_253_13.csv', 
              'TOI129.01_95_1.csv', 'TOI1341.01_237_3.csv', 'TOI2197.01_517_10.csv', 'TOI507.01_93_0.csv', 'TOI656.01_92_15.csv', 'TOI2464.01_252_3.csv', 'TOI1874.01_482_8.csv', 'TOI905.01_183_11.csv', 
              'TOI626.01_403_15.csv', 'TOI744.01_251_3.csv', 'TOI1458.01_352_19.csv', 'TOI241.01_247_0.csv', 'TOI1173.01_264_2.csv', 'TOI4381.01_1744_12.csv', 'TOI559.01_527_19.csv', 'TOI231.01_127_12.csv', 
              'TOI1779.01_106_3.csv', 'TOI511.01_353_4.csv', 'TOI2129.01_94_11.csv', 'TOI2014.01_457_3.csv', 'TOI640.01_336_17.csv', 'TOI1059.01_633_4.csv', 'TOI1970.01_281_14.csv', 'TOI107.01_770_14.csv', 
              'TOI1076.01_184_1.csv', 'TOI1963.01_665_6.csv', 'TOI987.01_265_13.csv', 'TOI1092.01_434_19.csv', 'TOI1107.01_575_6.csv', 'TOI585.01_352_14.csv', 'TOI114.01_332_2.csv', 'TOI4606.01_753_12.csv', 
              'TOI1627.01_145550102_15.csv', 'TOI398.01_255_8.csv', 'TOI173.01_902_4.csv', 'TOI1299.01_1070_1.csv', 'TOI477.01_277_11.csv', 'TOI1465.01_81_19.csv', 'TOI185.01_231_2.csv', 'TOI966.01_427_2.csv', 
              'TOI2222.01_249_8.csv', 'TOI778.01_457_2.csv', 'TOI481.01_642_6.csv', 'TOI1161.01_497_1.csv', 'TOI645.01_627_15.csv', 'TOI102.01_403_7.csv', 'TOI622.01_468_14.csv', 'TOI3501.01_195_6.csv', 
              'TOI573.01_230_1.csv', 'TOI1257.01_526_1.csv', 'TOI192.01_242_0.csv', 'TOI165.01_319_5.csv', 'TOI1176.01_388_17.csv', 'TOI1181.01_297_15.csv', 'TOI4140.01_380_6.csv', 'TOI246.01_440_8.csv', 
              'TOI2218.01_80_7.csv', 'TOI1337.01_511_7.csv', 'TOI413.01_853087145_8.csv', 'TOI1104.01_1077_1.csv', 'TOI811.01_257_7.csv', 'TOI501.01_179_15.csv', 'TOI984.01_633_13.csv', 'TOI1721.01_445_3.csv', 
              'TOI236.01_586_10.csv', 'TOI190.01_595_5.csv', 'TOI483.01_594_9.csv', 'TOI1163.01_204_0.csv', 'TOI964.01_141_8.csv', 'TOI1815.01_527_1.csv', 'TOI4612.01_768_4.csv', 'TOI404.01_255_14.csv', 
              'TOI101.01_139_8.csv', 'TOI4420.01_367_1.csv', 'TOI250.01_258_9.csv', 'TOI1864.01_219_17.csv', 'TOI567.01_181_0.csv', 'TOI828.01_234_17.csv', 'TOI590.01_13114_6.csv', 'TOI858.01_472_16.csv', 
              'TOI1976.01_798_5.csv', 'TOI495.01_356_0.csv', 'TOI1182.01_3849_3.csv', 'TOI116.01_221_10.csv', 'TOI3460.01_715_7.csv', 'TOI621.01_252_0.csv', 'TOI587.01_824_0.csv']
#p_csvlist = ['TOI267.01_735_1.csv']
#p_csvlist = ['TOI4470.01_0.csv']
df = pd.read_csv('./exofop_tess_tois.csv')
df = df[df['Planet SNR']>100]
df['TOI'] = df['TOI'].astype(str)
#ここはファイル名を要素にしたリストでfor loop
p_csv = p_csvlist[1]
#dataの呼び出し
TOInumber, _, _ = p_csv.split('_')
#TOInumber, _ = p_csv.split('_')
param_df = df[df['TOI'] == TOInumber[3:]]
duration = param_df['Duration (hours)'].values / 24
csvfile = f'./folded_lc_data/{TOInumber}.csv'
try:
    folded_table = ascii.read(csvfile)
except FileNotFoundError:
    sys.exit()
folded_lc = lk.LightCurve(data=folded_table)
folded_lc = folded_lc[(folded_lc.time.value < duration*0.8) & (folded_lc.time.value > -duration*0.8)]
#import astropy.units as u
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
#pdic['theta'] = np.pi
#mcmc_df.at['theta', 'vary_flags'] = False
mcmc_df = mcmc_df[mcmc_df['vary_flags']==True]
mcmc_params = mcmc_df.index.tolist()
for try_n in range(5):
    ###generate initial value for theta, phi
    mcmc_df.at['theta', 'output_value'] = np.arcsin(mcmc_df.at['b', 'output_value']/mcmc_df.at['a_rs', 'output_value'])+0.5
    mcmc_df.at['phi', 'output_value'] = np.random.uniform(0.0,np.pi)
    mcmc_pvalues = mcmc_df['output_value'].values
    print('mcmc_params: ', mcmc_params)
    print('mcmc_pvalues: ', mcmc_pvalues)
    df_for_mcmc.at['theta', 'mins'] = np.arcsin(mcmc_df.at['b', 'output_value']/mcmc_df.at['a_rs', 'output_value'])+0.49
    df_for_mcmc.at['theta', 'maxes'] = np.arcsin(mcmc_df.at['b', 'output_value']/mcmc_df.at['a_rs', 'output_value'])+0.51
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
    with Pool(processes=4) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(t, flux_data, flux_err_data.mean(), mcmc_params), pool=pool)
        sampler.run_mcmc(pos, max_n, progress=True)
        
    #sampler.reset()
    '''
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
    plt.plot(n, n / 50.0, "--k")
    plt.plot(n, y)
    plt.xlim(0, n.max())
    plt.ylim(0, y.max() + 0.1 * (y.max() - y.min()))
    plt.xlabel("number of steps")
    plt.ylabel(r"mean $\hat{\tau}$")
    plt.savefig(f'./mcmc_result/figure/{TOInumber}/tau_{try_n}.png')\
    ##plt.show()
    plt.close()
    print(tau)'''
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
    ###ライトカーブとモデルのプロット
    plt.errorbar(t, flux_data, yerr=flux_err_data, fmt=".k", capsize=0, alpha=0.1)
    for ind in inds:
        sample = flat_samples[ind]
        flux_model = ring_model(t, pdic, sample)
        plt.plot(t, flux_model, "C1", alpha=0.5)
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
    

    ###ポンチ絵の作成とcsvへの保存
    for ind in inds:
        sample = flat_samples[ind]
        flux_model = ring_model(t, pdic, sample)
        ###csvに書き出し###
        mcmc_res_df = mcmc_df
        mcmc_res_df['output_value'] = sample
        rp_rs = mcmc_res_df.at['rp_rs', 'output_value']
        rin_rp = mcmc_res_df.at['r_in', 'output_value']
        rout_rin = mcmc_res_df.at['r_out', 'output_value']
        b = mcmc_res_df.at['b', 'output_value']
        theta = mcmc_res_df.at['theta', 'output_value']
        phi = mcmc_res_df.at['phi', 'output_value']
        chi_square = np.sum(((flux_model-flux_data)/flux_err_data)**2)
        file_name = f'{TOInumber}_{chi_square:.0f}_{try_n}.pdf'
        plot_ring(rp_rs, rin_rp, rout_rin, b, theta, phi, file_name)
        os.makedirs(f'./mcmc_result/fit_pdata/{TOInumber}', exist_ok=True)
        mcmc_res_df.to_csv(f'./mcmc_result/fit_pdata/{TOInumber}/{TOInumber}_{chi_square:.0f}_{try_n}.csv', header=True, index=False)

'''fitting_data/data上で実行し、最もカイ2乗の小さいパラメータcsvを取得
TOIs = os.listdir()
list = []
for TOI in TOIs:
     os.chdir(TOI)
     files = os.listdir()
     file = sorted(files, key=natural_keys)[0]
     list.append(file)
     os.chdir('..')

p_full_csv_list = ['TOI3612.01_139_1.csv', 'TOI453.01_593_5.csv', 'TOI157.01_166_10.csv', 'TOI1796.01_143_17.csv', 'TOI1265.01_376_13.csv', 'TOI423.01_320_0.csv', 'TOI1134.01_876_10.csv', 'TOI2126.01_116_0.csv', 
              'TOI1292.01_185_4.csv', 'TOI1478.01_241_8.csv', 'TOI1937.01_132_6.csv', 'TOI2131.01_371_9.csv', 'TOI924.01_216_0.csv', 'TOI677.01_221_0.csv', 'TOI1825.01_2750_3.csv', 'TOI4059.01_158_0.csv', 
              'TOI527.01_321_3.csv', 'TOI2140.01_232_9.csv', 'TOI1456.01_1226_8.csv', 'TOI2579.01_381_8.csv', 'TOI1615.01_252_5.csv', 'TOI1936.01_282_16.csv', 'TOI199.01_674_17.csv', 'TOI4623.01_226_11.csv', 
              'TOI508.01_398_7.csv', 'TOI818.01_164_8.csv', 'TOI1431.01_284_19.csv', 'TOI2127.01_693_16.csv', 'TOI1135.01_2505_13.csv', 'TOI1264.01_43612_5.csv', 'TOI820.01_352_10.csv', 'TOI1951.01_299_11.csv', 
              'TOI1833.01_488_12.csv', 'TOI1670.01_1786_3.csv', 'TOI822.01_390_17.csv', 'TOI1025.01_426_12.csv', 'TOI1454.01_464_0.csv', 'TOI1826.01_125_7.csv', 'TOI212.01_317_7.csv', 'TOI143.01_362_5.csv', 
              'TOI2020.01_445_6.csv', 'TOI1150.01_2398_17.csv', 'TOI683.01_207_8.csv', 'TOI418.01_145_2.csv', 'TOI766.01_253_0.csv', 'TOI845.01_530_17.csv', 'TOI1271.01_2046_12.csv', 'TOI1934.01_554_14.csv', 
              'TOI1270.01_141_4.csv', 'TOI1312.01_386_0.csv', 'TOI738.01_196_1.csv', 'TOI767.01_301_10.csv', 'TOI1425.01_321_3.csv', 'TOI1455.01_204_2.csv', 'TOI834.01_315_3.csv', 'TOI4138.01_375_12.csv', 
              'TOI1151.01_473_2.csv', 'TOI3846.01_171_1.csv', 'TOI2021.01_254_5.csv', 'TOI675.01_755_9.csv', 'TOI1830.01_1305_14.csv', 'TOI391.01_213_17.csv', 'TOI4516.01_152_12.csv', 'TOI2154.01_196_3.csv', 
              'TOI1248.01_158_4.csv', 'TOI349.01_324_9.csv', 'TOI2025.01_304_18.csv', 'TOI671.01_487_4.csv', 'TOI1823.01_5979_3.csv', 'TOI267.01_735_1.csv', 'TOI1612.01_223_18.csv', 'TOI1283.01_450_7.csv', 
              'TOI1274.01_245_8.csv', 'TOI121.01_282_9.csv', 'TOI1767.01_545_7.csv', 'TOI857.01_462_14.csv', 'TOI1050.01_398_11.csv', 'TOI774.01_120_6.csv', 'TOI4470.01_720_7.csv', 'TOI1682.01_453_0.csv', 
              'TOI1419.01_138_3.csv', 'TOI1909.01_273_6.csv', 'TOI2200.01_261_7.csv', 'TOI201.01_3670_0.csv', 'TOI4486.01_901_11.csv', 'TOI150.01_564_9.csv', 'TOI1845.01_160_4.csv', 'TOI1766.01_499_2.csv', 
              'TOI1295.01_303_16.csv', 'TOI1300.01_365_6.csv', 'TOI934.01_166_0.csv', 'TOI4449.01_135_2.csv', 'TOI1771.01_491_8.csv', 'TOI1420.01_233_2.csv', 'TOI433.01_165_1.csv', 'TOI1019.01_439_6.csv', 
              'TOI4087.01_283_16.csv', 'TOI4162.01_439_0.csv', 'TOI2119.01_257_7.csv', 'TOI899.01_270_8.csv', 'TOI1069.01_198_0.csv', 'TOI2024.01_381_0.csv', 'TOI147.01_196_7.csv', 'TOI264.01_337_2.csv', 
              'TOI135.01_452_19.csv', 'TOI1676.01_89_3.csv', 'TOI1714.01_394_13.csv', 'TOI758.01_739_5.csv', 'TOI1141.01_5997_10.csv', 'TOI665.01_594_5.csv', 'TOI1385.01_3345_3.csv', 'TOI479.01_196_1.csv', 
              'TOI780.01_205_9.csv', 'TOI1302.01_389_9.csv', 'TOI1297.01_307_10.csv', 'TOI615.01_444_2.csv', 'TOI1130.01_280_10.csv', 'TOI781.01_283_2.csv', 'TOI1198.01_1589_14.csv', 'TOI272.01_351_6.csv', 
              'TOI1924.01_548_11.csv', 'TOI123.01_194036_14.csv', 'TOI1259.01_157_0.csv', 'TOI842.01_128_4.csv', 'TOI182.01_189_3.csv', 'TOI224.01_49685_0.csv', 'TOI1810.01_166_2.csv', 'TOI1651.01_286_4.csv', 
              'TOI471.01_259_18.csv', 'TOI1268.01_325_2.csv', 'TOI105.01_380_1.csv', 'TOI959.01_181_3.csv', 'TOI3849.01_211_7.csv', 'TOI625.01_259_4.csv', 'TOI112.01_261_15.csv', 'TOI195.01_264_10.csv', 
              'TOI1186.01_1826_4.csv', 'TOI655.01_129_0.csv', 'TOI194.01_421_0.csv', 'TOI232.01_193_3.csv', 'TOI1647.01_1287_3.csv', 'TOI1725.01_146_1.csv', 'TOI163.01_360_0.csv', 'TOI2000.01_330_9.csv', 
              'TOI769.01_317_1.csv', 'TOI490.01_328_17.csv', 'TOI1012.01_204_12.csv', 'TOI505.01_367_0.csv', 'TOI624.01_847_16.csv', 'TOI1251.01_308_6.csv', 'TOI3492.01_334_7.csv', 'TOI4601.01_334_0.csv', 
              'TOI1148.01_3492_1.csv', 'TOI1861.01_8245_15.csv', 'TOI368.01_278_0.csv', 'TOI1494.01_136_1.csv', 'TOI1811.01_171_4.csv', 'TOI1236.01_282_2.csv', 'TOI2017.01_438_14.csv', 'TOI1573.01_651_1.csv', 
              'TOI2403.01_428_11.csv', 'TOI4545.01_216_2.csv', 'TOI159.01_438_7.csv', 'TOI106.01_401_9.csv', 'TOI3960.01_207_0.csv', 'TOI1165.01_181_5.csv', 'TOI472.01_112_9.csv', 'TOI4535.01_253_13.csv', 
              'TOI129.01_95_1.csv', 'TOI1341.01_237_3.csv', 'TOI2197.01_517_10.csv', 'TOI507.01_93_0.csv', 'TOI656.01_92_15.csv', 'TOI2464.01_252_3.csv', 'TOI1874.01_482_8.csv', 'TOI905.01_183_11.csv', 
              'TOI626.01_403_15.csv', 'TOI744.01_251_3.csv', 'TOI1458.01_352_19.csv', 'TOI241.01_247_0.csv', 'TOI1173.01_264_2.csv', 'TOI4381.01_1744_12.csv', 'TOI559.01_527_19.csv', 'TOI231.01_127_12.csv', 
              'TOI1779.01_106_3.csv', 'TOI511.01_353_4.csv', 'TOI2129.01_94_11.csv', 'TOI2014.01_457_3.csv', 'TOI640.01_336_17.csv', 'TOI1059.01_633_4.csv', 'TOI1970.01_281_14.csv', 'TOI107.01_770_14.csv', 
              'TOI1076.01_184_1.csv', 'TOI1963.01_665_6.csv', 'TOI987.01_265_13.csv', 'TOI1092.01_434_19.csv', 'TOI1107.01_575_6.csv', 'TOI585.01_352_14.csv', 'TOI114.01_332_2.csv', 'TOI4606.01_753_12.csv', 
              'TOI1627.01_145550102_15.csv', 'TOI398.01_255_8.csv', 'TOI173.01_902_4.csv', 'TOI1299.01_1070_1.csv', 'TOI477.01_277_11.csv', 'TOI1465.01_81_19.csv', 'TOI185.01_231_2.csv', 'TOI966.01_427_2.csv', 
              'TOI2222.01_249_8.csv', 'TOI778.01_457_2.csv', 'TOI481.01_642_6.csv', 'TOI1161.01_497_1.csv', 'TOI645.01_627_15.csv', 'TOI102.01_403_7.csv', 'TOI622.01_468_14.csv', 'TOI3501.01_195_6.csv', 
              'TOI573.01_230_1.csv', 'TOI1257.01_526_1.csv', 'TOI192.01_242_0.csv', 'TOI165.01_319_5.csv', 'TOI1176.01_388_17.csv', 'TOI1181.01_297_15.csv', 'TOI4140.01_380_6.csv', 'TOI246.01_440_8.csv', 
              'TOI2218.01_80_7.csv', 'TOI1337.01_511_7.csv', 'TOI413.01_853087145_8.csv', 'TOI1104.01_1077_1.csv', 'TOI811.01_257_7.csv', 'TOI501.01_179_15.csv', 'TOI984.01_633_13.csv', 'TOI1721.01_445_3.csv', 
              'TOI236.01_586_10.csv', 'TOI190.01_595_5.csv', 'TOI483.01_594_9.csv', 'TOI1163.01_204_0.csv', 'TOI964.01_141_8.csv', 'TOI1815.01_527_1.csv', 'TOI4612.01_768_4.csv', 'TOI404.01_255_14.csv', 
              'TOI101.01_139_8.csv', 'TOI4420.01_367_1.csv', 'TOI250.01_258_9.csv', 'TOI1864.01_219_17.csv', 'TOI567.01_181_0.csv', 'TOI828.01_234_17.csv', 'TOI590.01_13114_6.csv', 'TOI858.01_472_16.csv', 
              'TOI1976.01_798_5.csv', 'TOI495.01_356_0.csv', 'TOI1182.01_3849_3.csv', 'TOI116.01_221_10.csv', 'TOI3460.01_715_7.csv', 'TOI621.01_252_0.csv', 'TOI587.01_824_0.csv']
'''