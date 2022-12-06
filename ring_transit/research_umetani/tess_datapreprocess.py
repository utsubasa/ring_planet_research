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
import pickle

warnings.filterwarnings("ignore")


def calc_data_survival_rate(lc, duration):
    data_n = len(lc.flux)
    # mid_transit_timeからdurationの前後×2.5 [min]/ 2 min cadence
    max_data_n = (duration * 5 * 60 * 24 / 2)
    data_survival_rate = data_n / max_data_n
    print(f"{data_survival_rate:2f}% data usable")

    return data_survival_rate

def _make_diff_bool(list_1, list_2):
    """2つのリストの共通しない要素についてbool値を出力

    Args:
        list_1 (List): first list to diff
        list_2 (List): second list to diff
    """
    diff = list_1 - list_2
    diff_bool = ~(diff==0)
    return diff_bool


def calc_obs_transit_time(t0list, t0errlist, transit_time_list, transit_time_error):
    """
    return estimated period or cleaned light curve
    mid transit time+period*iのリストと実際にフィットしたリストの差分をとる
    diff=0は解析していないepochを表しているので除外している。
    Attributes:
        t0list(List):実際にフィッティングした結果得たt0のリスト
        t0errlist(List):実際にフィッティングした結果得たt0のエラーのリスト
        transit_time_list(List):mid transit time+period*iのリスト
        transit_time_error(float):EXOFOPのtableにあるtransit_time_errorの値
    """
    diff_bool = _make_diff_bool(t0list, transit_time_list)
    transit_time_list = transit_time_list[diff_bool]
    t0list = np.array(t0list)[diff_bool]
    t0errlist = np.array(t0errlist)[diff_bool]
    x = np.array(t0list)
    y = np.array(x - transit_time_list) * 24  # [days] > [hours]
    yerr = (np.sqrt(np.square(t0errlist) + np.square(transit_time_error)) * 24)  # [days] > [hours]
    if save_res == True:
        os.makedirs(f"{homedir}/SAP_fitting_result/data/calc_obs_transit_time/{poly_type}/", exist_ok=True,)
        
        pd.DataFrame({"x": x, "O-C": y, "yerr": yerr}).to_csv(
            f"{homedir}/SAP_fitting_result/data/calc_obs_transit_time/{poly_type}/{TOInumber}.csv"
        )
        
        plt.errorbar(x=x, y=y, yerr=yerr, fmt=".k")
        plt.xlabel("mid transit time[BJD] - 2457000")
        plt.ylabel("O-C(hrs)")
        plt.tight_layout()
        os.makedirs(
            f"{homedir}/SAP_fitting_result/figure/calc_obs_transit_time/{poly_type}/", exist_ok=True,
        )
        plt.savefig(f"{homedir}/SAP_fitting_result/figure/calc_obs_transit_time/{poly_type}/{TOInumber}.png")
        plt.close()

    x = np.arange(len(t0list))
    y = np.array(t0list)
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
        if save_res == True:
            os.makedirs(
                f"{homedir}/SAP_fitting_result/figure/estimate_period/{poly_type}/", exist_ok=True,
            )
            plt.savefig(f"{homedir}/SAP_fitting_result/figure/estimate_period/{poly_type}/{TOInumber}.png")
        # plt.show()
        plt.close()
        return estimated_period, ts * res.stderr
    else:
        print("np.isnan(ts*res.stderr) == True")
        pdb.set_trace()
        estimated_period = period
        return estimated_period

def clip_outliers(
    res,
    lc,
    outliers,
    t0list,
    t0errlist,
    folded_lc=False,
    transit_and_poly_fit=False,
    process=None,
):
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
                if save_res == True:
                    os.makedirs(
                        f"{homedir}/SAP_fitting_result/data/folded_lc/outliers/{poly_type}/",
                        exist_ok=True,
                    )
                    outliers.write(f"{homedir}/SAP_fitting_result/data/folded_lc/outliers/{poly_type}/{TOInumber}.csv")
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
                if save_res == True:
                    os.makedirs(
                        f"{homedir}/SAP_fitting_result/figure/each_lc/transit_fit/{poly_type}/{TOInumber}",
                        exist_ok=True,
                    )
                    
                    plt.savefig(
                        f"{homedir}/SAP_fitting_result/figure/each_lc/transit_fit/{poly_type}/{TOInumber}/{TOInumber}_{str(i)}.png",
                        header=False,
                        index=False,
                    )
                
            else:
                if save_res == True:
                    os.makedirs(
                        f"{homedir}/SAP_fitting_result/figure/each_lc/{poly_type}/{process}/{TOInumber}",
                        exist_ok=True,
                    )
                    
                    plt.savefig(
                        f"{homedir}/SAP_fitting_result/figure/each_lc/{poly_type}/{process}/{TOInumber}/{TOInumber}_{str(i)}.png",
                        header=False,
                        index=False,
                    )
                
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
    lc.time = lc.time - res.params["t0"].value
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

    if poly_type == '4poly':
        model = lmfit.models.PolynomialModel(degree=4)
        poly_params = model.make_params(c0=1, c1=0, c2=0, c3=0, c4=0)
        poly_params['c4'].set(max=0.1, min=-0.1)
    elif poly_type == '3poly':
        model = lmfit.models.PolynomialModel(degree=3)
        poly_params = model.make_params(c0=1, c1=0, c2=0, c3=0)
    elif poly_type == '2poly':
        model = lmfit.models.PolynomialModel(degree=2)
        poly_params = model.make_params(c0=1, c1=0, c2=0)

    result = model.fit(
        out_transit.flux.value, poly_params, x=out_transit.time.value
    )
    result.plot()
    if save_res == True:
        os.makedirs(
            f"{homedir}/SAP_fitting_result/figure/curvefit/{poly_type}/{TOInumber}", exist_ok=True
        )
        plt.savefig(f"{homedir}/SAP_fitting_result/figure/curvefit/{poly_type}/{TOInumber}/{TOInumber}_{str(i)}.png")
    plt.close()

    return result


def curvefit_normalize(each_lc, poly_params):
    if poly_type == '4poly':
        poly_model = np.polynomial.Polynomial(
            [
                poly_params["c0"].value,
                poly_params["c1"].value,
                poly_params["c2"].value,
                poly_params["c3"].value,
                poly_params["c4"].value,
            ]
        )
    elif poly_type == '3poly':
        poly_model = np.polynomial.Polynomial(
            [
                poly_params["c0"].value,
                poly_params["c1"].value,
                poly_params["c2"].value,
                poly_params["c3"].value,
            ]
        )
    elif poly_type == '2poly':
        poly_model = np.polynomial.Polynomial(
            [
                poly_params["c0"].value,
                poly_params["c1"].value,
                poly_params["c2"].value,
            ]
        )


    # normalization
    each_lc.flux = each_lc.flux.value / poly_model(each_lc.time.value)
    each_lc.flux_err = each_lc.flux_err.value / poly_model(each_lc.time.value)
    each_lc.errorbar()
    if save_res == True:
        os.makedirs(
            f"{homedir}/SAP_fitting_result/figure/each_lc/after_curvefit/{poly_type}/{TOInumber}",
            exist_ok=True,
        )
        plt.savefig(f"{homedir}/SAP_fitting_result/figure/each_lc/after_curvefit/{poly_type}/{TOInumber}/{TOInumber}_{str(i)}.png")
    plt.close()

    return each_lc


def folding_lc_from_csv(TOInumber, loaddir, process):
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
            process=None,
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
    if save_res == True:
        os.makedirs(
            f"/Users/u_tsubasa/Dropbox/ring_planet_research/SAP_folded_lc/figure/{poly_type}/{process}",
            exist_ok=True,
        )
        plt.savefig(f"/Users/u_tsubasa/Dropbox/ring_planet_research/SAP_folded_lc/figure/{poly_type}/{process}/{TOInumber}.png")
    #plt.show()
    plt.close()
    if save_res == True:
        os.makedirs(
            f"{homedir}/SAP_fitting_result/data/folded_lc/{poly_type}/{process}/csv",
            exist_ok=True,
        )
        cleaned_lc.write(f"{homedir}/SAP_fitting_result/data/folded_lc/{poly_type}/{process}/csv/{TOInumber}.csv")

    return res


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
    if poly_type == '4poly':
        poly_model = np.polynomial.Polynomial(
            [
                poly_params["c0"],
                poly_params["c1"],
                poly_params["c2"],
                poly_params["c3"],
                poly_params["c4"],
            ]
        )
    elif poly_type == '3poly':
        poly_model = np.polynomial.Polynomial(
            [
                poly_params["c0"],
                poly_params["c1"],
                poly_params["c2"],
                poly_params["c3"],
            ]
        )
    elif poly_type == '2poly':
        poly_model = np.polynomial.Polynomial(
            [
                poly_params["c0"],
                poly_params["c1"],
                poly_params["c2"],
            ]
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


def transit_fitting(
    lc,
    rp_rs,
    period,
    fitting_model=no_ring_transitfit,
    transitfit_params=None,
    curvefit_params=None,
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
                if poly_type == '4poly':
                    params.add_many(
                    curvefit_params["c0"],
                    curvefit_params["c1"],
                    curvefit_params["c2"],
                    curvefit_params["c3"],
                    curvefit_params["c4"],
                    )   
                elif poly_type == '3poly':
                    params.add_many(
                    curvefit_params["c0"],
                    curvefit_params["c1"],
                    curvefit_params["c2"],
                    curvefit_params["c3"],
                    )
                elif poly_type == '2poly':
                    params.add_many(
                    curvefit_params["c0"],
                    curvefit_params["c1"],
                    curvefit_params["c2"],
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
            ax = lc.scatter()
            flux_model = no_ring_transitfit(res.params, lc.time.value, lc.flux.value, lc.flux_err.value, p_names, return_model=True)
            ax.plot(lc.time.value, flux_model, label="fitting model", color="black")
            plt.show()
            print(lmfit.fit_report(res))
            pdb.set_trace()
    res = sorted(best_res_dict.items())[0][1]
    print(f"reduced chisquare: {res.redchi:4f}")
    return res


def q_to_u_limb(q_arr):
    q1 = q_arr[0]
    q2 = q_arr[1]
    u1 = np.sqrt(q1) * 2 * q2
    u2 = np.sqrt(q1) * (1 - 2 * q2)
    return np.array([u1, u2])

homedir = ("/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani")

oridf = pd.read_csv(f"{homedir}/exofop_tess_tois_2022-09-13.csv")
df = oridf[oridf["Planet SNR"] > 100]
with open('./no_data_found_toi.txt', 'rb') as f:
    no_data_found_list = pickle.load(f)  # short がないか、SPOCがないか
no_perioddata_list = [1134.01, 1897.01, 2423.01, 2666.01, 4465.01,]  # exofopの表にperiodの記載無し。1567.01,1656.01もperiodなかったがこちらはcadence=’short’のデータなし。
no_signal_list = [2218.01]  # トランジットのsignalが無いか、ノイズに埋もれて見えない
multiplanet_list = [1670.01, 201.01, 822.01]  # , 1130.01]
startrend_list = [4381.01, 1135.01, 1025.01, 212.01, 1830.01, 2119.01, 224.01]
flare_list = [212.01, 1779.01, 2119.01]
two_epoch_list = [
    671.01,1963.01,1283.01,758.01,1478.01,3501.01,845.01,121.01,1104.01,811.01,3492.01,1312.01,1861.01,665.01,224.01,2047.01,5379.01,5149.01,
    5518.01,5640.01,319.01,2783.01,5540.01,1840.01,5686.01,5226.01,937.01,4725.01,4731.01,5148.01, 1130.01]
ignore_list = [
    1130.01,
    224.01,
    123.01,
    1823.01,
    1292.01,
    2218.01,
    964.01,
    1186.01,
    1296.01,
    1254.01,
    2351.01,
    1070.01,
    1344.01

]
duration_ng = [129.01, 182.01, 1059.01, 1182.01, 1425.01, 1455.01, 1811.01,2154.01, 3910.01] 
trend_ng = [1069.01, 1092.01, 1141.01, 1163.01, 1198.01, 1270.01, 1299.01, 1385.01, 1454.01, 1455.01, 1647.01, 1796.01,]
fold_ng = [986.01, 1041.01]
# 既に前処理したTOIの重複した前処理を回避するためのTOIのリスト
done4poly_list = os.listdir("/Users/u_tsubasa/Dropbox/ring_planet_research/SAP_folded_lc/figure/4poly/obs_t0")
done4poly_list = [s for s in done4poly_list if "TOI" in s]
done4poly_list = [s.lstrip("TOI") for s in done4poly_list]
done4poly_list = [float(s.strip(".png")) for s in done4poly_list]
done4poly_list = [float(s) for s in done4poly_list]

#df = df.sort_values("Planet SNR", ascending=False)

"""処理を行わないTOIを選択する"""
df = df.set_index(["TOI"])
#df = df.drop(index=done4poly_list, errors="ignore")
df = df.drop(index=no_data_found_list, errors="ignore")
# df = df.drop(index=multiplanet_list, errors='ignore')
df = df.drop(index=no_perioddata_list, errors="ignore")
# df = df.drop(index=startrend_list, errors='ignore')
df = df.drop(index=flare_list, errors="ignore")
df = df.drop(index=two_epoch_list, errors="ignore")
df = df.drop(index=ignore_list, errors="ignore")
#df = df.drop(index=duration_ng, errors='ignore')
#df = df.drop(index=fold_ng, errors='ignore')
#df = df.drop(index=trend_ng, errors='ignore')
df = df.reset_index()

df["TOI"] = df["TOI"].astype(str)
TOIlist = df["TOI"]
"""
hole_lc_list = os.listdir('{homedir}/SAP_fitting_result/hole_lc_plot')
calc_t0_2ndloop_list = os.listdir('/Users/u_tsubasa/Dropbox/ring_planet_research/folded_lc/figure/obs_t0')
sym_diff = set(hole_lc_list) ^ set(calc_t0_2ndloop_list)
print(list(sym_diff))
import pdb;pdb.set_trace()
"""

poly_type = '4poly'
save_res = False
for TOI in [4470.01, ]:
    print("analysing: ", "TOI" + str(TOI))
    TOI = str(TOI)
    """惑星、主星の各パラメータを取得"""
    param_df = df[df["TOI"] == TOI]
    TOInumber = "TOI" + str(param_df["TOI"].values[0])
    duration = param_df["Duration (hours)"].values[0] / 24
    period = param_df["Period (days)"].values[0]
    transit_time = (param_df["Transit Epoch (BJD)"].values[0] - 2457000.0)  # translate BTJD
    transit_time_error = param_df["Transit Epoch error"].values[0]
    rp = (param_df["Planet Radius (R_Earth)"].values[0] * 0.00916794)  # translate to Rsun
    rs = param_df["Stellar Radius (R_Sun)"].values[0]
    rp_rs = rp / rs

    """もしもduration, period, transit_timeどれかのパラメータがnanだったらそのTOIを記録して、処理はスキップする
    if np.sum(np.isnan([duration, period, transit_time])) != 0:
        if save_res == True:
            with open(f'nan3params_toi_{poly_type}.dat', 'a') as f:
                f.write(f'{TOInumber}: {np.isnan([duration, period, transit_time])}\n')
        else:
            pass
        continue
    """
    """多惑星系の場合、ターゲットのトランジットに影響があるかを判断する。
    print('judging whether other planet transit is included in the data...')
    other_p_df = oridf[oridf['TIC ID'] == param_df['TIC ID'].values[0]]
    if len(other_p_df.index) != 1:
        if save_res == True:
            with open(f'multiplanet_toi_{poly_type}.dat', 'a') as f:
                f.write(f'{TOInumber}\n')
        else:
            pass
        continue
        #lc = remove_others_transit(lc, oridf, param_df, other_p_df, TOI)
    """

    """lightkurveを用いてSPOCが作成した2min cadenceの全セクターのライトカーブをダウンロードする """
    search_result = lk.search_lightcurve(
        f"TOI {TOI[:-3]}", mission="TESS", cadence="short", author="SPOC"
    )
    lc_collection = search_result.download_all()
    #lc_collection = search_result[0].download()
    """全てのライトカーブを結合し、fluxがNaNのデータ点は除去する"""
    lc = lc_collection.stitch().remove_nans()  # initialize lc
    #lc = lc_collection.remove_nans()  # initialize lc
    lc.flux = lc.sap_flux
    lc.flux_err = lc.sap_flux_err
    
    #lc.scatter()
    #plt.savefig(f'{homedir}/hole_lc_plot/TOI{TOI}.png')
    #plt.close()
    #import pdb;pdb.set_trace()
    
    """ターゲットの惑星のtransit time listを作成"""
    transit_time_list = np.append(
        np.arange(transit_time, lc.time[-1].value, period),
        np.arange(transit_time, lc.time[0].value, -period),
    )
    transit_time_list = np.unique(transit_time_list)
    transit_time_list.sort()

    
    """各エポックで外れ値除去と多項式フィッティング"""
    # 値を格納するリストの定義
    outliers = []
    t0list = []
    t0errlist = []
    
    #ax = lc.scatter()
    for i, mid_transit_time in enumerate(transit_time_list):
        print(f"preprocessing...epoch: {i}")
        """
        ax.axvline(x=mid_transit_time)
        continue
        """
        """トランジットの中心時刻から±duration*2.5の時間帯を切り取る"""
        epoch_start = mid_transit_time - (duration * 2.5)
        epoch_end = mid_transit_time + (duration * 2.5)
        tmp = lc[lc.time.value > epoch_start]
        each_lc = tmp[tmp.time.value < epoch_end]
        each_lc = (
            each_lc.fold(period=period, epoch_time=mid_transit_time)
            .normalize()
            .remove_nans()
        )

        """データ点が理論値の90%未満ならそのエポックは解析対象から除外"""
        data_survival_rate = calc_data_survival_rate(each_lc, duration)
        if data_survival_rate < 0.9:
            if data_survival_rate != 0.0:
                ax = each_lc.errorbar()
                ax.set_title(f"{data_survival_rate:4f} useable")
                if save_res == True:
                    os.makedirs(
                        f"{homedir}/SAP_fitting_result/figure/error_lc/under_90%_data/calc_t0/{poly_type}/{TOInumber}",
                        exist_ok=True,
                    )
                    plt.savefig(f"{homedir}/SAP_fitting_result/figure/error_lc/under_90%_data/calc_t0/{poly_type}/{TOInumber}/{TOInumber}_{str(i)}.png")
                plt.close()
            t0list.append(mid_transit_time)
            t0errlist.append(np.nan)
            continue

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
                process="1stloop",
            )
            if len(outliers) == 0:
                if save_res == True:
                    os.makedirs(
                        f"{homedir}/SAP_fitting_result/data/each_lc/{poly_type}/calc_t0/{TOInumber}",
                        exist_ok=True,
                    )
                    each_lc.write(f"{homedir}/SAP_fitting_result/data/each_lc/{poly_type}/calc_t0/{TOInumber}/{TOInumber}_{str(i)}.csv")
                    os.makedirs(
                    f"{homedir}/SAP_fitting_result/data/each_lc/fit_statistics/1stloop/{poly_type}/{TOInumber}",
                    exist_ok=True,
                    )
                    with open(
                        f"{homedir}/SAP_fitting_result/data/each_lc/fit_statistics/1stloop/{poly_type}/{TOInumber}/{TOInumber}_{str(i)}.txt",
                        "a",
                    ) as f:
                        print(lmfit.fit_report(transit_res), file=f)
                break
            
        else:
            pass
    #plt.show()
    #import pdb;pdb.set_trace()
    
    """t0list, t0errlistのpickle化"""
    t0_dict = {'t0list': t0list, 't0errlist': t0errlist}
    os.makedirs(f"{homedir}/SAP_fitting_result/data/t0dicts/{poly_type}", exist_ok=True)
    if os.path.exists(f"{homedir}/SAP_fitting_result/data/t0dicts/{poly_type}/{TOInumber}.pkl") == False:
        with open(f"{homedir}/SAP_fitting_result/data/t0dicts/{poly_type}/{TOInumber}.pkl", 'wb') as f:
            pickle.dump(t0_dict, f)

    """transit epochが2つ以下のTOIを記録しておく
    if len(t0list) <= 2:
        with open("two_period_toi.dat", "a") as f:
            f.write(f"{TOI}\n")
    """
    
    
    """folded_lcに対してtransit fitting & remove outliers. transit parametersを得る"""
    print("folding and calculate duration...")
    time.sleep(1)
    fold_res = folding_lc_from_csv(
        TOInumber,
        loaddir=f"{homedir}/SAP_fitting_result/data/each_lc/{poly_type}/calc_t0",
        process="calc_t0",
    )
    
    
    """各エポックでのtransit fittingで得たmid_transit_timeのリストからorbital period、durationを算出"""
    #with open(f"{homedir}/SAP_fitting_result/data/t0dicts/{poly_type}/{TOInumber}.pkl", 'rb') as f:
        #t0_dict = pickle.load(f)

    t0list = t0_dict['t0list']
    t0errlist = t0_dict['t0errlist']
    period, period_err = calc_obs_transit_time(t0list, t0errlist, transit_time_list, transit_time_error)
    # _, _ = calc_obs_transit_time(t0list, t0errlist, transit_time_list, transit_time_error)
    a_rs = fold_res.params["a"].value
    b = fold_res.params["b"].value
    inc = np.arccos(b / a_rs)
    if np.isnan(rp_rs):
        rp_rs = fold_res.params["rp"].value
    duration = (period / np.pi) * np.arcsin(
        (1 / a_rs)
        * (np.sqrt(np.square(1 + rp_rs) - np.square(b)) / np.sin(inc))
    )
    
    """transit fittingによって得たtransit time, period, durationを記録"""
    if save_res == True:
        obs_t0_idx = np.abs(np.asarray(t0list) - transit_time).argmin()
        os.makedirs(
            f"{homedir}/SAP_fitting_result/data/folded_lc/{poly_type}/calc_t0/fit_statistics/{TOInumber}",
            exist_ok=True,
        )
        with open(
            f"{homedir}/SAP_fitting_result/data/folded_lc/{poly_type}/calc_t0/fit_statistics/{TOInumber}/{TOInumber}_folded.txt",
            "a",
        ) as f:
            print(lmfit.fit_report(fold_res), file=f)
            print(f"calculated duration[day]: {duration}", file=f)
            print(f"obs_transit_time[day]: {t0list[obs_t0_idx]}", file=f)
            print(f"obs_transit_time_err[day]: {t0errlist[obs_t0_idx]}", file=f)
            print(f"obs_period[day]: {period}", file=f)
            print(f"obs_period_err[day]: {period_err}", file=f)
    
    """fittingで得たtransit time listを反映"""
    obs_t0_list = t0list
    outliers = []
    t0list = []
    t0errlist = []











    #duration = param_df["Duration (hours)"].values[0] / 24
    """transit parametersをfixして、baseline,t0を決める"""
    print('2nd loop')
    for i, mid_transit_time in enumerate(obs_t0_list):
    # for i, mid_transit_time in enumerate(transit_time_list):
        print(f"reprocessing...epoch: {i}")
        """
        if i == 33:
            t0list.append(mid_transit_time)
            t0errlist.append(np.nan)
            continue
        """
        
        """トランジットの中心時刻からduration*2.5の時間帯を切り取る"""
        epoch_start = mid_transit_time - (duration * 2.5)
        epoch_end = mid_transit_time + (duration * 2.5)
        tmp = lc[lc.time.value > epoch_start]
        each_lc = tmp[tmp.time.value < epoch_end]
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
                if save_res == True:
                    os.makedirs(
                        f"{homedir}/SAP_fitting_result/figure/error_lc/under_90%_data/obs_t0/{poly_type}/{TOInumber}",
                        exist_ok=True,
                    )
                    plt.savefig(f"{homedir}/SAP_fitting_result/figure/error_lc/under_90%_data/obs_t0/{poly_type}/{TOInumber}/{TOInumber}_{str(i)}.png")
                plt.close()
            t0list.append(mid_transit_time)
            t0errlist.append(np.nan)
            continue

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
            each_lc, outliers, t0list, t0errlist = clip_outliers(
                res,
                each_lc,
                outliers,
                t0list,
                t0errlist,
                transit_and_poly_fit=True,
                process="2ndloop",
            )
            each_lc = curvefit_normalize(each_lc, res.params)
            if len(outliers) == 0:
                if save_res == True:
                    os.makedirs(
                        f"{homedir}/SAP_fitting_result/data/each_lc/{poly_type}/obs_t0/{TOInumber}",
                        exist_ok=True,
                    )
                    each_lc.write(f"{homedir}/SAP_fitting_result/data/each_lc/{poly_type}/obs_t0/{TOInumber}/{TOInumber}_{str(i)}.csv")
                    os.makedirs(
                        f"{homedir}/SAP_fitting_result/data/each_lc/fit_statistics/2ndloop/{poly_type}/{TOInumber}",
                        exist_ok=True,
                    )
                    with open(
                        f"{homedir}/SAP_fitting_result/data/each_lc/fit_statistics/2ndloop/{poly_type}/{TOInumber}/{TOInumber}_{str(i)}.txt",
                        "a",
                    ) as f:
                        print(lmfit.fit_report(res), file=f)
                break
            else:
                pass
        # 処理したeach_lcは指定されたディレクトリに保存される。

    """最終的なfolded_lcを生成する。"""
    print("refolding...")
    time.sleep(1)
    fold_res = folding_lc_from_csv(
        TOInumber,
        loaddir=f"{homedir}/SAP_fitting_result/data/each_lc/{poly_type}/obs_t0/",
        process="obs_t0",
    )
    # fold_res = folding_lc_from_csv(TOInumber, loaddir=f'{homedir}/SAP_fitting_result/data/each_lc/calc_t0', process='calc_t0_2ndloop')
    os.makedirs(
        f"{homedir}/SAP_fitting_result/data/folded_lc/{poly_type}/obs_t0/fit_statistics/{TOInumber}",
        exist_ok=True,
    )
    with open(
        f"{homedir}/SAP_fitting_result/data/folded_lc/{poly_type}/obs_t0/fit_statistics/{TOInumber}/{TOInumber}_folded.txt",
        "a",
    ) as f:
        print(lmfit.fit_report(fold_res), file=f)
    print(f"Analysis completed: {TOInumber}")

# import pdb;pdb.set_trace()
