# -*- coding: utf-8 -*-
# import astropy.units as u
import os
import pdb
import time
import warnings

import batman
import lightkurve as lk
import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.table import vstack
from scipy.stats import linregress, t
import c_compile_ring

warnings.filterwarnings("ignore")

def calc_data_survival_rate(lc, duration):
    data_n = len(lc.flux)
    max_data_n = (
        duration * 5 * 60 * 24 / 2
    )  # mid_transit_timeからdurationの前後×2.5 [min]/ 2 min cadence
    data_survival_rate = data_n / max_data_n
    print(f"{data_survival_rate:2f}% data usable")
    # max_data_n = (lc.time_original[-1]-lc.time_original[0])*24*60/2
    return data_survival_rate


def q_to_u_limb(q_arr):
    q1 = q_arr[0]
    q2 = q_arr[1]
    u1 = np.sqrt(q1) * 2 * q2
    u2 = np.sqrt(q1) * (1 - 2 * q2)
    return np.array([u1, u2])


def set_params_batman(params_lm, p_names, limb_type="quadratic"):
    params = batman.TransitParams()  # object to store transit parameters
    params.limb_dark = limb_type  # limb darkening model
    q_arr = np.zeros(2)
    for i in range(len(p_names)):
        value = params_lm[p_names[i]]
        name = p_names[i]
        if name == "t0":
            params.t0 = value
        if name == "per":
            params.per = value
        if name == "rp":
            params.rp = value
        if name == "a":
            params.a = value
        # if name=="inc":
        # params.inc = value
        if name == "b":
            params.inc = np.degrees(np.arccos(value / params.a))
        if name == "ecc":
            params.ecc = value
        if name == "w":
            params.w = value
        if name == "q1":
            q_arr[0] = value
        if name == "q2":
            q_arr[1] = value
    u_arr = q_to_u_limb(q_arr)
    params.u = u_arr
    return params


def set_params_lm(p_names, values, mins, maxes, vary_flags):
    params = lmfit.Parameters()
    for i in range(len(p_names)):
        if vary_flags[i]:
            params.add(
                p_names[i],
                value=values[i],
                min=mins[i],
                max=maxes[i],
                vary=vary_flags[i],
            )
        else:
            params.add(p_names[i], value=values[i], vary=vary_flags[i])
    return params


# リングなしモデルをfitting
def no_ring_transitfit(params, x, data, eps_data, p_names, return_model=False):
    global chi_square
    params_batman = set_params_batman(params, p_names)
    m = batman.TransitModel(params_batman, x)  # initializes model
    model = m.light_curve(params_batman)  # calculates light curve
    chi_square = np.sum(((data - model) / eps_data) ** 2)
    # print(params)
    # print(chi_square)
    if return_model == True:
        return model
    else:
        return (data - model) / eps_data


def no_ring_transit_and_polynomialfit(
    params, x, data, eps_data, p_names, return_model=False
):
    global chi_square
    params_batman = set_params_batman(params, p_names)
    m = batman.TransitModel(params_batman, x)  # initializes model
    transit_model = m.light_curve(params_batman)  # calculates light curve
    poly_params = params.valuesdict()
    poly_model = np.polynomial.Polynomial(
        [poly_params["c0"], poly_params["c1"], poly_params["c2"], poly_params["c3"], poly_params["c4"]]
    )
    polynomialmodel = poly_model(x)
    model = transit_model + polynomialmodel - 1
    # chi_square = np.sum(((data-model)/eps_data)**2)
    # print(params)
    # print(chi_square)
    if return_model == True:
        return model, transit_model, polynomialmodel
    else:
        return (data - model) / eps_data


def transit_params_setting(rp_rs, period):
    global p_names
    """トランジットフィッティングパラメータの設定"""
    p_names = ["t0", "per", "rp", "a", "b", "ecc", "w", "q1", "q2"]
    if np.isnan(rp_rs):
        values = [
            np.random.uniform(-0.05, 0.05),
            period,
            np.random.uniform(0.05, 0.1),
            np.random.uniform(1, 10),
            np.random.uniform(0.0, 0.5),
            0,
            90.0,
            np.random.uniform(0.1, 0.9),
            np.random.uniform(0.1, 0.9),
        ]
        maxes = [0.5, period * 1.2, 0.5, 1000, 1.1, 0.8, 90, 1.0, 1.0]
    else:
        values = [
            np.random.uniform(-0.05, 0.05),
            period,
            rp_rs,
            np.random.uniform(1, 10),
            np.random.uniform(0.0, 0.5),
            0,
            90.0,
            np.random.uniform(0.1, 0.9),
            np.random.uniform(0.1, 0.9),
        ]
        maxes = [0.5, period * 1.2, 0.5, 1000, 1 + rp_rs, 0.8, 90, 1.0, 1.0]
    mins = [-0.5, period * 0.8, 0.01, 1, 0, 0, 90, 0.0, 0.0]
    vary_flags = [True, False, True, True, True, False, False, True, True]
    return set_params_lm(p_names, values, mins, maxes, vary_flags)


def calc_obs_transit_time(
    t0list, t0errlist, num_list, transit_time_list, transit_time_error
):
    """
    return estimated period or cleaned light curve
    """
    
    diff = t0list - transit_time_list
    transit_time_list = transit_time_list[~(diff == 0)]
    t0errlist = t0errlist[~(t0errlist == 0)]
    x = np.array(t0list)[~(diff == 0)]
    y = np.array(x - transit_time_list) * 24  # [days] > [hours]
    yerr = (
        np.sqrt(np.square(t0errlist) + np.square(transit_time_error)) * 24
    )  # [days] > [hours]
    pd.DataFrame({"x": x, "O-C": y, "yerr": yerr}).to_csv(
        f"{homedir}/fitting_result/data/calc_obs_transit_time/{TOInumber}.csv"
    )
    plt.errorbar(x=x, y=y, yerr=yerr, fmt=".k")
    plt.xlabel("mid transit time[BJD] - 2457000")
    plt.ylabel("O-C(hrs)")
    plt.tight_layout()
    os.makedirs(
                f"{homedir}/fitting_result/figure/simulation_TOI495.01/plussin/calc_obs_transit_time/",
                exist_ok=True,
                )
    plt.savefig(
        f"{homedir}/fitting_result/figure/simulation_TOI495.01/plussin/calc_obs_transit_time/{TOInumber}.png"
    )
    plt.close()
    
    x = np.array(num_list)
    y = np.array(t0list)[~(diff == 0)]
    yerr = t0errlist
    try:
        res = linregress(x, y)
    except ValueError:
        print("ValueError: Inputs must not be empty.")
        pdb.set_trace()
    estimated_period = res.slope
    tinv = lambda p, df: abs(t.ppf(p / 2, df))
    ts = tinv(0.05, len(x) - 2)

    if np.isnan(ts * res.stderr) == False:
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)  # for plotting residuals
        try:
            ax1.errorbar(x=x, y=y, yerr=yerr, fmt=".k")
        except TypeError:
            ax1.scatter(x=x, y=y, color="r")
        ax1.plot(x, res.intercept + res.slope * x, label="fitted line")
        ax1.text(
            0.5,
            0.2,
            f"period: {res.slope:.6f} +/- {ts*res.stderr:.6f}",
            transform=ax1.transAxes,
        )
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("mid transit time[BJD] - 2457000")
        residuals = y - (res.intercept + res.slope * x)
        try:
            ax2.errorbar(x=x, y=residuals, yerr=yerr, fmt=".k")
        except TypeError:
            ax2.scatter(x=x, y=residuals, color="r")
        ax2.plot(x, np.zeros(len(x)), color="red")
        ax2.set_xlabel("mid transit time[BJD] - 2457000")
        ax2.set_ylabel("residuals")
        plt.tight_layout()
        os.makedirs(
                f"{homedir}/fitting_result/figure/simulation_TOI495.01/plussin/estimate_period/",
                exist_ok=True,
                )
        plt.savefig(
            f"{homedir}/fitting_result/figure/simulation_TOI495.01/plussin/estimate_period/{TOInumber}.png"
        )
        # plt.show()
        plt.close()
        return estimated_period, ts * res.stderr
    else:
        print("np.isnan(ts*res.stderr) == True")
        pdb.set_trace()
        estimated_period = period
        return estimated_period


def transit_fitting(
    lc,
    rp_rs,
    period,
    fitting_model=no_ring_transitfit,
    transitfit_params=None,
    curvefit_params=None
):
    """transit fitting"""
    flag_time = np.abs(lc.time.value) < 1.0
    lc = lc[flag_time]
    t = lc.time.value
    flux = lc.flux.value
    flux_err = lc.flux_err.value
    best_res_dict = {}  # 最も良いreduced chi-squareを出した結果を選別し保存するための辞書
    while len(best_res_dict) == 0:
        for _ in range(30):
            params = transit_params_setting(rp_rs, period)
            if transitfit_params != None:
                for p_name in p_names:
                    params[p_name].set(value=transitfit_params[p_name].value)
                    if p_name != "t0":
                        params[p_name].set(vary=False)
                    else:
                        #t0の初期値だけ動かし、function evalsの少なくてエラーが計算できないことを回避
                        params[p_name].set(value=np.random.uniform(-0.05, 0.05))
            if curvefit_params != None:
                params.add_many(
                    curvefit_params["c0"],
                    curvefit_params["c1"],
                    curvefit_params["c2"],
                    curvefit_params["c3"],
                    curvefit_params["c4"],
                )
            res = lmfit.minimize(
                fitting_model,
                params,
                args=(t, flux, flux_err, p_names),
                max_nfev=10000,
            )
            if res.params["t0"].stderr != None:
                if np.isfinite(res.params["t0"].stderr):
                    # and res.redchi < 10:
                    # if res.redchi < 10:
                    red_redchi = res.redchi - 1
                    best_res_dict[red_redchi] = res
        if len(best_res_dict) == 0:
            print(TOInumber, i)
            lc.scatter()
            plt.show()
            print(lmfit.fit_report(res))
            pdb.set_trace()
    res = sorted(best_res_dict.items())[0][1]
    print(f"reduced chisquare: {res.redchi:4f}")
    return res


def clip_outliers(res, lc, outliers, t0list, t0errlist, folded_lc=False, transit_and_poly_fit=False, savedir=None):
    t = lc.time.value
    flux = lc.flux.value
    flux_err = lc.flux_err.value
    if transit_and_poly_fit == True:
        (
            flux_model,
            transit_model,
            polynomial_model,
        ) = no_ring_transit_and_polynomialfit(
            res.params, t, flux, flux_err, p_names, return_model=True
        )
    else:
        flux_model = no_ring_transitfit(
            res.params, t, flux, flux_err, p_names, return_model=True
        )

    residual_lc = lc.copy()
    residual_lc.flux = np.sqrt(np.square(flux_model - lc.flux))
    _, mask = residual_lc.remove_outliers(sigma=5.0, return_mask=True)
    inverse_mask = np.logical_not(mask)

    if np.all(inverse_mask) == True:
        print("no outliers")
        if folded_lc == True:
            try:
                outliers = vstack(outliers)
                os.makedirs(
                    "/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/fitting_result/data/simulation_TOI495.01/plussin/folded_lc/outliers",
                    exist_ok=True,
                )
                outliers.write(
                    f"/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/fitting_result/data/simulation_TOI495.01/plussin/folded_lc/outliers/{TOInumber}.csv"
                )
            except ValueError:
                pass
            outliers = []
            return lc, outliers, t0list, t0errlist
        else:
            fig = plt.figure()
            ax1 = fig.add_subplot(
                2, 1, 1
            )  # for plotting transit model and data
            ax2 = fig.add_subplot(2, 1, 2)  # for plotting residuals
            lc.errorbar(ax=ax1, color="gray", marker=".", alpha=0.3)
            ax1.plot(t, flux_model, label="fitting model", color="black")
            if transit_and_poly_fit == True:
                ax1.plot(
                    t,
                    transit_model,
                    label="transit model",
                    ls="--",
                    color="blue",
                    alpha=0.5,
                )
                ax1.plot(
                    t,
                    polynomial_model,
                    label="polynomial model",
                    ls="-.",
                    color="red",
                    alpha=0.5,
                )

            try:
                outliers = vstack(outliers)
                outliers.errorbar(
                    ax=ax1, color="cyan", label="outliers(each_lc)", marker="."
                )
            except ValueError:
                pass

            ax1.legend()
            ax1.set_title(f"chi square/dof: {int(res.chisqr)}/{res.nfree} ")
            residuals = lc - flux_model
            residuals.errorbar(ax=ax2, color="gray", marker=".")
            ax2.plot(t, np.zeros(len(t)), label="fitting model", color="black")
            ax2.set_ylabel("residuals")
            plt.tight_layout()

            if transit_and_poly_fit == False:
                os.makedirs(
                    f"{homedir}/fitting_result/figure/simulation_TOI495.01/plussin/each_lc/transit_fit/{TOInumber}",
                    exist_ok=True,
                )
                plt.savefig(
                    f"{homedir}/fitting_result/figure/simulation_TOI495.01/plussin/each_lc/transit_fit/{TOInumber}/{TOInumber}_{str(i)}.png",
                    header=False,
                    index=False,
                )
                # plt.savefig(f'{homedir}/fitting_result/figure/each_lc/bls/{TOInumber}/{TOInumber}_{str(i)}.png', header=False, index=False)
            else:
                os.makedirs(
                    f"{homedir}/fitting_result/figure/simulation_TOI495.01/plussin/each_lc/{savedir}/{TOInumber}",
                    exist_ok=True,
                )
                plt.savefig(
                    f"{homedir}/fitting_result/figure/simulation_TOI495.01/plussin/each_lc/{savedir}/{TOInumber}/{TOInumber}_{str(i)}.png",
                    header=False,
                    index=False,
                )
                # plt.savefig(f'{homedir}/fitting_result/figure/each_lc/bls/{TOInumber}/{TOInumber}_{str(i)}.png', header=False, index=False)
            plt.close()
            t0list.append(res.params["t0"].value + mid_transit_time)
            t0errlist.append(res.params["t0"].stderr)
            outliers = []
            lc.time = lc.time - res.params["t0"].value
    else:
        print("outliers exist")
        # print('removed bins:', len(each_lc[mask]))
        outliers.append(lc[mask])
        lc = lc[~mask]
    plt.close()
    return lc, outliers, t0list, t0errlist


def curve_fitting(each_lc, duration, res=None):
    if res != None:
        out_transit = each_lc[
            (each_lc["time"].value < res.params["t0"].value - (duration * 0.7))
            | (
                each_lc["time"].value
                > res.params["t0"].value + (duration * 0.7)
            )
        ]
    else:
        out_transit = each_lc[
            (each_lc["time"].value < -(duration * 0.7))
            | (each_lc["time"].value > (duration * 0.7))
        ]
    model = lmfit.models.PolynomialModel(degree=4)
    # poly_params = model.make_params(c0=0, c1=0, c2=0, c3=0, c4=0, c5=0, c6=0, c7=0)
    poly_params = model.make_params(c0=1, c1=0, c2=0, c3=0, c4=0)
    result = model.fit(
        out_transit.flux.value, poly_params, x=out_transit.time.value
    )
    result.plot()
    os.makedirs(
        f"{homedir}/fitting_result/figure/simulation_TOI495.01/plussin/curvefit/{TOInumber}", exist_ok=True
    )
    plt.savefig(
        f"{homedir}/fitting_result/figure/simulation_TOI495.01/plussin/curvefit/{TOInumber}/{TOInumber}_{str(i)}.png"
    )
    # os.makedirs(f'{homedir}/fitting_result/figure/curvefit/bls/{TOInumber}', exist_ok=True)
    # plt.savefig(f'{homedir}/fitting_result/figure/curvefit/bls/{TOInumber}/{TOInumber}_{str(i)}.png')
    plt.close()

    return result


def curvefit_normalize(each_lc, poly_params):
    poly_model = np.polynomial.Polynomial(
        [
            poly_params["c0"].value,
            poly_params["c1"].value,
            poly_params["c2"].value,
            poly_params["c3"].value,
            poly_params["c4"].value,
        ]
    )
    # normalization
    each_lc.flux = each_lc.flux.value / poly_model(each_lc.time.value)
    each_lc.flux_err = each_lc.flux_err.value / poly_model(each_lc.time.value)
    
    return each_lc


def folding_lc_from_csv(TOInumber, loaddir, savedir):
    outliers = []
    t0list = []
    t0errlist = []
    each_lc_list = []
    try:
        each_lc_list = []
        total_lc_csv = os.listdir(f"{loaddir}/{TOInumber}/")
        total_lc_csv = [i for i in total_lc_csv if "TOI" in i]
        for each_lc_csv in total_lc_csv:
            each_table = ascii.read(f"{loaddir}/{TOInumber}/{each_lc_csv}")
            each_lc = lk.LightCurve(data=each_table)
            each_lc_list.append(each_lc)
    except ValueError:
        pass
    cleaned_lc = vstack(each_lc_list)
    cleaned_lc.sort("time")

    while True:
        res = transit_fitting(
            cleaned_lc, rp_rs, period, fitting_model=no_ring_transitfit
        )
        cleaned_lc, outliers, _, _ = clip_outliers(
            res,
            cleaned_lc,
            outliers,
            t0list,
            t0errlist,
            folded_lc=True,
            transit_and_poly_fit=False,
            savedir=None,
        )
        if len(outliers) == 0:
            break
    flux_model = no_ring_transitfit(
        res.params,
        cleaned_lc.time.value,
        cleaned_lc.flux.value,
        cleaned_lc.flux_err.value,
        p_names,
        return_model=True,
    )
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)  # for plotting transit model and data
    ax2 = fig.add_subplot(2, 1, 2)  # for plotting residuals
    cleaned_lc.errorbar(
        ax=ax1, color="black", marker=".", zorder=1, label="data"
    )
    ax1.plot(
        cleaned_lc.time.value,
        flux_model,
        label="fitting model",
        color="red",
        zorder=2,
    )
    ax1.legend()
    ax1.set_title(TOInumber)
    residuals = cleaned_lc - flux_model
    residuals.errorbar(
        ax=ax2, color="black", ecolor="gray", alpha=0.3, marker=".", zorder=1
    )
    ax2.plot(
        cleaned_lc.time.value,
        np.zeros(len(cleaned_lc.time)),
        color="red",
        zorder=2,
    )
    ax2.set_ylabel("residuals")
    plt.tight_layout()
    os.makedirs(
        f"/Users/u_tsubasa/Dropbox/ring_planet_research/folded_lc/figure/simulation_TOI495.01/plussin/{savedir}",
        exist_ok=True,
    )
    plt.savefig(
        f"/Users/u_tsubasa/Dropbox/ring_planet_research/folded_lc/figure/simulation_TOI495.01/plussin/{savedir}/{TOInumber}.png"
    )
    # plt.show()
    plt.close()
    os.makedirs(
        f"/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/fitting_result/data/simulation_TOI495.01/plussin/folded_lc/{savedir}",
        exist_ok=True,
    )
    cleaned_lc.write(
        f"/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/fitting_result/data/simulation_TOI495.01/plussin/folded_lc/{savedir}/{TOInumber}.csv"
    )

    return res


def make_simulation_data(mid_transit_time):
    """make_simulation_data"""
    t = np.linspace(-0.3, 0.3, 500)
    names = ["q1", "q2", "t0", "porb", "rp_rs", "a_rs",
            "b", "norm", "theta", "phi", "tau", "r_in",
            "r_out", "norm2", "norm3", "ecosw", "esinw"]

    saturnlike_values = [0.26, 0.36, np.random.uniform(-1e-3,1e-3), 1.27, 0.123, 3.81,
            0.10, 1, np.pi/6.74, 0, 1, 1.53,
            1.95, 0.0, 0.0, 0.0, 0.0]

    mins = [0.0, 0.0, -0.1, 0.0, 0.01, 1.0,
            0.0, 0.9, 0.0, 0.0, 0.0, 1.00,
            1.01, -0.1, -0.1, 0.0, 0.0]

    maxes = [1.0, 1.0, 0.1, 100.0, 0.5, 100.0,
            1.0, 1.1, np.pi, np.pi, 1.0, 2.45,
            2.45, 0.1, 0.1, 0.0, 0.0]

    vary_flags = [True, True, False, False, True, True,
                True, False, True, True, False, True,
                True, False, False, False, False]

    ###土星likeなTOI495.01のパラメータで作成したモデル###
    pdic_saturnlike = dict(zip(names, saturnlike_values))
    #ymodel = ring_model(t, pdic_saturnlike)*(1+(np.sin( (t/0.6 +1.2*np.random.rand()) * np.pi)*0.01)) + np.random.randn(len(t))*0.001
    ymodel = ring_model(t, pdic_saturnlike) + np.random.randn(len(t))*0.001 + np.sin( (t/0.6 +1.2*np.random.rand()) * np.pi)*0.01
    yerr = np.array(t/t)*1e-3
    each_lc = lk.LightCurve(t, ymodel, yerr)
    each_lc.time = each_lc.time + mid_transit_time + np.random.randn()*0.01
    plt.errorbar(each_lc.time.value, each_lc.flux.value, each_lc.flux_err.value, fmt='.k')
    #plt.show()
    os.makedirs(
                f"{homedir}/fitting_result/figure/simulation_TOI495.01/plussin/before_process/each_lc",
                exist_ok=True,
            )
    plt.savefig(f"{homedir}/fitting_result/figure/simulation_TOI495.01/plussin/before_process/each_lc/{i}.png")
    #plt.show()
    plt.close()
    os.makedirs(
                f"{homedir}/fitting_result/data/simulation_TOI495.01/plussin/before_process/each_lc",
                exist_ok=True,
            )
    each_lc.write(f"{homedir}/fitting_result/data/simulation_TOI495.01/plussin/before_process/each_lc/{i}.csv")

    return each_lc


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

def save_each_lc(each_lc, savedir):
    each_lc.errorbar()
    os.makedirs(
        f"{homedir}/fitting_result/figure/simulation_TOI495.01/plussin/each_lc/after_curvefit/{TOInumber}",
        exist_ok=True,
    )
    plt.savefig(
        f"{homedir}/fitting_result/figure/simulation_TOI495.01/plussin/each_lc/after_curvefit/{TOInumber}/{TOInumber}_{str(i)}.png"
    )
    plt.close()
    os.makedirs(
        f"{homedir}/fitting_result/data/simulation_TOI495.01/plussin/each_lc/{savedir}/{TOInumber}",
        exist_ok=True,
    )
    each_lc.write(
        f"{homedir}/fitting_result/data/simulation_TOI495.01/plussin/each_lc/{savedir}/{TOInumber}/{TOInumber}_{str(i)}.csv"
    )

homedir = (
    "/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani"
)

oridf = pd.read_csv(f"{homedir}/exofop_tess_tois.csv")
df = oridf[oridf["Planet SNR"] > 100]
df = df.sort_values("Planet SNR", ascending=False)
df["TOI"] = df["TOI"].astype(str)
TOIlist = df["TOI"]

for TOI in [495.01]:
    # for TOI in TOIlist:
    print("analysing: ", "TOI" + str(TOI))
    TOI = str(TOI)
    # if TOI=='1823.01' or TOI=='1833.01' or TOI=='2218.01' or TOI=='224.01':
    # continue

    """惑星、主星の各パラメータを取得"""
    param_df = df[df["TOI"] == TOI]
    TOInumber = "TOI" + str(param_df["TOI"].values[0])
    duration = param_df["Duration (hours)"].values[0] / 24
    period = param_df["Period (days)"].values[0]
    transit_time = (
        param_df["Transit Epoch (BJD)"].values[0] - 2457000.0
    )  # translate BTJD
    transit_time_error = param_df["Transit Epoch error"].values[0]
    rp = (
        param_df["Planet Radius (R_Earth)"].values[0] * 0.00916794
    )  # translate to Rsun
    rs = param_df["Stellar Radius (R_Sun)"].values[0]
    rp_rs = rp / rs

    """各エポックで外れ値除去と多項式フィッティング"""
    
    # 値を格納するリストの定義
    EPOCH_NUM = 54

    outliers = []
    t0list = []
    t0errlist = []
    num_list = np.arange(EPOCH_NUM)
    transit_time_list = []
    # ax = lc.scatter()
    #for i, mid_transit_time in enumerate(transit_time_list):
    #for i in [0,]:
    for i in range(EPOCH_NUM):
        print(f"preprocessing...epoch: {i}")
        mid_transit_time = transit_time + period*i
        transit_time_list.append(mid_transit_time)
        each_lc = make_simulation_data(mid_transit_time)
        #each_table = ascii.read(f"{homedir}/fitting_result/data/simulation_TOI495.01/plussin/before_process/each_lc/{i}.csv")
        #each_lc = lk.LightCurve(data=each_table)
        
        each_lc = (
            each_lc.fold(period=period, epoch_time=mid_transit_time)
            .normalize()
            .remove_nans()
        )


        """外れ値除去と多項式フィッティングを外れ値が検知されなくなるまで繰り返す"""
        while True:
            transit_res = transit_fitting(each_lc, rp_rs, period)
            curvefit_res = curve_fitting(each_lc, duration, transit_res)
            each_lc = curvefit_normalize(each_lc, curvefit_res.params)
            transit_res = transit_fitting(each_lc, rp_rs, period)
            each_lc, outliers, t0list, t0errlist = clip_outliers(
                transit_res,
                each_lc,
                outliers,
                t0list,
                t0errlist,
                transit_and_poly_fit=False,
                savedir="1stloop",
            )
            if len(outliers) == 0:
                save_each_lc(each_lc, "calc_t0")
                break

        os.makedirs(
            f"{homedir}/fitting_result/data/simulation_TOI495.01/plussin/each_lc/modelresult/1stloop/{TOInumber}",
            exist_ok=True,
        )
        with open(
            f"{homedir}/fitting_result/data/simulation_TOI495.01/plussin/each_lc/modelresult/1stloop/{TOInumber}/{TOInumber}_{str(i)}.txt",
            "a",
        ) as f:
            print(lmfit.fit_report(transit_res), file=f)
    # plt.show()
    #import pdb;pdb.set_trace()
    """folded_lcに対してtransit fitting & remove outliers. transit parametersを得る"""
    print("folding and calculate duration...")
    time.sleep(1)
    fold_res = folding_lc_from_csv(
        TOInumber,
        #loaddir=f"{homedir}/fitting_result/data/each_lc/calc_t0",
        loaddir=f"{homedir}/fitting_result/data/simulation_TOI495.01/plussin/each_lc/calc_t0",
        savedir="calc_t0_1stloop",
    )
    transit_time_list = np.array(transit_time_list)
    #_, _ = calc_obs_transit_time(t0list, t0errlist, num_list, transit_time_list, transit_time_error)
    """fittingで得たtransit time listを反映"""
    obs_t0_list = t0list
    outliers = []
    t0list = []
    t0errlist = []
    num_list = []
    
    """transit parametersをfixして、baseline,t0を決める"""
    for i, mid_transit_time in enumerate(obs_t0_list):
    #for i in range(54):
        print(f"reprocessing...epoch: {i}")
        each_table = ascii.read(f"{homedir}/fitting_result/data/simulation_TOI495.01/plussin/before_process/each_lc/{i}.csv")
        each_lc = lk.LightCurve(data=each_table)
        each_lc = (
            each_lc.fold(period=period, epoch_time=mid_transit_time)
            .normalize()
            .remove_nans()
        )
        """解析中断条件を満たさないかチェック"""
        data_survival_rate = calc_data_survival_rate(each_lc, duration)
        if data_survival_rate < 0.9:
            if data_survival_rate != 0.0:
                ax = each_lc.errorbar()
                ax.set_title(f"{data_survival_rate:4f} useable")
                os.makedirs(
                    f"{homedir}/fitting_result/figure/simulation_TOI495.01/plussin/error_lc/under_90%_data/obs_t0/{TOInumber}",
                    exist_ok=True,
                )
                plt.savefig(
                    f"{homedir}/fitting_result/figure/simulation_TOI495.01/plussin/error_lc/under_90%_data/obs_t0/{TOInumber}/{TOInumber}_{str(i)}.png"
                )
                plt.close()
            t0list.append(mid_transit_time)
            t0errlist.append(np.nan)
            continue
        else:
            num_list.append(i)

        while True:
            transit_res = transit_fitting(each_lc, rp_rs, period)
            curvefit_res = curve_fitting(each_lc, duration, transit_res)
            res = transit_fitting(
                each_lc,
                rp_rs,
                period,
                fitting_model=no_ring_transit_and_polynomialfit,
                transitfit_params=fold_res.params,
                curvefit_params=curvefit_res.params,
            )
            os.makedirs(
                f"{homedir}/fitting_result/data/simulation_TOI495.01/plussin/each_lc/modelresult/2ndloop/{TOInumber}",
                exist_ok=True,
            )
            with open(
                f"{homedir}/fitting_result/data/simulation_TOI495.01/plussin/each_lc/modelresult/2ndloop/{TOInumber}/{TOInumber}_{str(i)}.txt",
                "a",
            ) as f:
                print(lmfit.fit_report(res), file=f)
            each_lc, outliers, t0list, t0errlist = clip_outliers(
                res,
                each_lc,
                outliers,
                t0list,
                t0errlist,
                transit_and_poly_fit=True,
                savedir="2ndloop",
            )
            each_lc = curvefit_normalize(each_lc, res.params)
            if len(outliers) == 0:
                save_each_lc(each_lc, "obs_t0")
                break
                
            else:
                pass

    """最終的なfolded_lcを生成する。"""
    print("refolding...")
    time.sleep(1)
    fold_res = folding_lc_from_csv(
        TOInumber,
        loaddir=f"{homedir}/fitting_result/data/simulation_TOI495.01/plussin/each_lc/obs_t0/",
        savedir="obs_t0",
    )
    # fold_res = folding_lc_from_csv(TOInumber, loaddir=f'{homedir}/fitting_result/data/each_lc/calc_t0', savedir='calc_t0_2ndloop')
    os.makedirs(
        f"{homedir}/fitting_result/data/simulation_TOI495.01/plussin/folded_lc/modelresult/2ndloop/{TOInumber}",
        exist_ok=True,
    )
    with open(
        f"{homedir}/fitting_result/data/simulation_TOI495.01/plussin/folded_lc/modelresult/2ndloop/{TOInumber}/{TOInumber}_folded.txt",
        "a",
    ) as f:
        print(lmfit.fit_report(fold_res), file=f)
    print(f"Analysis completed: {TOInumber}")
    each_lc.scatter()
    plt.title(f"Analysis completed: {TOInumber}")
    plt.show()

# import pdb;pdb.set_trace()
