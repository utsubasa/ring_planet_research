# -*- coding: utf-8 -*-
import os
import pdb
import sys
import warnings

import batman
import lightkurve as lk
import lmfit
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from astropy.io import ascii
from numpy.linalg import svd
from scipy import integrate

import c_compile_ring

warnings.filterwarnings("ignore")


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


def set_params_lm(names, values, mins, maxes, vary_flags):
    params = lmfit.Parameters()
    for i in range(len(names)):
        if vary_flags[i]:
            params.add(
                names[i],
                value=values[i],
                min=mins[i],
                max=maxes[i],
                vary=vary_flags[i],
            )
        else:
            params.add(names[i], value=values[i], vary=vary_flags[i])
    return params


# Ring model
# Input "x" (1d array), "pdic" (dic)
# Ouput flux (1d array)
def ring_model(t, pdic, mcmc_pvalues=None):
    if mcmc_pvalues is None:
        pass
    else:
        for i, param in enumerate(mcmc_params):
            # print(i, v[i])
            pdic[param] = mcmc_pvalues[i]

    q1, q2, t0, porb, rp_rs, a_rs, b, norm = (
        pdic["q1"],
        pdic["q2"],
        pdic["t0"],
        pdic["porb"],
        pdic["rp_rs"],
        pdic["a_rs"],
        pdic["b"],
        pdic["norm"],
    )
    theta, phi, tau, r_in, r_out = (
        pdic["theta"],
        pdic["phi"],
        pdic["tau"],
        pdic["r_in"],
        pdic["r_out"],
    )
    norm2, norm3 = pdic["norm2"], pdic["norm3"]
    cosi = b / a_rs
    u = [2 * np.sqrt(q1) * q2, np.sqrt(q1) * (1 - 2 * q2)]
    u1, u2 = u[0], u[1]
    ecosw = pdic["ecosw"]
    esinw = pdic["esinw"]

    # rp_rs, theta, phi, r_in, r_out = p
    # theta, phi = p
    # rp_rs, theta, phi = p
    # rp_rs, theta, phi, r_in = p
    """ when e & w used: ecosw -> e, esinw -> w (deg)
    e = ecosw
    w = esinw*np.pi/180.0
    ecosw, esinw = e*np.cos(w), e*np.sin(w)
    """

    # see params_def.h
    pars = np.array(
        [
            porb,
            t0,
            ecosw,
            esinw,
            b,
            a_rs,
            theta,
            phi,
            tau,
            r_in,
            r_out,
            rp_rs,
            q1,
            q2,
        ]
    )
    times = np.array(t)
    # print(pars)
    # print(c_compile_ring.getflux(times, pars, len(times)))
    model_flux = np.array(c_compile_ring.getflux(times, pars, len(times))) * (
        norm + norm2 * (times - t0) + norm3 * (times - t0) ** 2
    )
    model_flux = np.nan_to_num(model_flux)
    return model_flux


# リングありモデルをfitting
def ring_transitfit(params, x, data, eps_data, p_names, return_model=False):
    # start =time.time()
    model = ring_model(x, params.valuesdict())
    chi_square = np.sum(((data - model) / eps_data) ** 2)
    # print(params)
    print(chi_square)
    # print(np.max(((data-model)/eps_data)**2))
    if return_model == True:
        return model
    else:
        return (data - model) / eps_data


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

    # ring outer & innter radii
    R_in = rp_rs * rin_rp
    R_out = rp_rs * rin_rp * rout_rin

    ## calculte of ellipse of rings
    a0 = 1
    b0 = -np.sin(phi) / np.tan(theta)
    c0 = (np.sin(phi) ** 2) / (np.tan(theta) ** 2) + (
        (np.cos(phi) ** 2) / (np.sin(theta) ** 2)
    )
    A = np.array([[a0, b0], [b0, c0]])
    u, s, vh = svd(A)
    angle = (np.arctan2(u[0][0], u[0][1])) * 180 / np.pi
    major = s[0]
    major_to_minor = np.sqrt((s[1] / s[0]))

    # plotter
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes()
    c = patches.Circle(xy=(0, 0), radius=1.0, fc="orange", ec="orange")
    c2 = patches.Circle(xy=(0, -b), radius=rp_rs, fc="k", ec="k")
    e = patches.Ellipse(
        xy=(0, -b),
        width=2 * R_in,
        height=2 * R_in * major_to_minor,
        angle=angle,
        fc="none",
        ec="r",
    )
    e2 = patches.Ellipse(
        xy=(0, -b),
        width=2 * R_out,
        height=2 * R_out * major_to_minor,
        angle=angle,
        fc="none",
        ec="r",
    )
    ax.add_patch(c)
    ax.add_patch(c2)
    ax.add_patch(e)
    ax.add_patch(e2)
    # ax.set_title(f'chisq={str(ring_res.redchi)[:6]}')
    plt.axis("scaled")
    ax.set_aspect("equal")
    # os.makedirs(f'./lmfit_result/illustration/{TOInumber}', exist_ok=True)
    # os.makedirs(f'./simulation/illustration/{TOInumber}', exist_ok=True)
    # plt.savefig(f'./simulation/illustration/{TOInumber}/{file_name}', bbox_inches="tight")
    plt.show()


def binning_lc(folded_lc):
    binned_flux, binned_time, _ = scipy.stats.binned_statistic(
        folded_lc.time.value, folded_lc.flux.value, bins=500
    )
    # まず、リスト内要素をひとつずらした新規リストを2つ作成する
    binned_time_list = list(binned_time)
    n = 1
    binned_time_list2 = (
        binned_time_list[n:] + binned_time_list[:n]
    )  # 要素のインデックスをひとつずらす
    binned_time_list.pop()  # リストの末尾の要素を削除
    binned_time_list2.pop()
    # 上記二つのリストを用いて、2つずつをひとつずらしずつ取り出す
    binned_flux_err = []
    for x1, x2 in zip(binned_time_list, binned_time_list2):
        mask = (x1 <= folded_lc["time"].value) & (folded_lc["time"].value < x2)
        flux_err = folded_lc[mask].flux_err.value
        if len(flux_err) == 0:
            binned_flux_err.append(np.nan)
        else:
            flux_err = np.sqrt(np.nansum(flux_err**2)) / len(flux_err)
            binned_flux_err.append(flux_err)

    return binned_time[1:], binned_flux, binned_flux_err


# CSVFILE = './folded_lc_data/TOI2403.01.csv'
# done_TOIlist = os.listdir('./lmfit_result/transit_fit') #ダブリ解析防止
oridf = pd.read_csv("./exofop_tess_tois.csv")
df = oridf[oridf["Planet SNR"] > 100]
df["TOI"] = df["TOI"].astype(str)
df = df.sort_values("Planet SNR", ascending=False)
df["TOI"] = df["TOI"].astype(str)
TOIlist = df["TOI"]
"""
plot_ring(
    rp_rs=0.123,
    rin_rp=1.01,
    rout_rin=1.70,
    b=0.0,
    theta=15 * np.pi / 180,
    phi=0 * np.pi / 180,
    file_name="test.png",
)
"""
for TOI in [495.01]:
    # for TOI in TOIlist[330:]:
    TOI = str(TOI)
    print(TOI)
    TOInumber = "TOI" + TOI
    param_df = df[df["TOI"] == TOI]
    duration = param_df["Duration (hours)"].values[0] / 24
    period = param_df["Period (days)"].values[0]

    # lmfit.minimizeのためのparamsのセッティング。これはリングありモデル
    ###parameters setting###
    TOInumber = "TOI" + str(param_df["TOI"].values[0])
    rp = param_df["Planet Radius (R_Earth)"].values[0] * 0.00916794
    # translate to Rsun
    rs = param_df["Stellar Radius (R_Sun)"].values[0]
    rp_rs = rp / rs

    CSVFILE = "/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/fitting_result/data/simulation_TOI495.01/plussin_plusmodel/45deg_0deg/folded_lc/obs_t0/TOI495.01.csv"
    # CSVFILE = f'/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/fitting_result/data/folded_lc/{TOInumber}.csv'
    try:
        folded_table = ascii.read(CSVFILE)
    except FileNotFoundError:
        # with open('CSVFILE_FileNotFoundError.txt', 'a') as f:
        # f.write(TOInumber+'\n')
        continue
    folded_lc = lk.LightCurve(data=folded_table)
    folded_lc = folded_lc[
        (folded_lc.time.value < duration * 0.7)
        & (folded_lc.time.value > -duration * 0.7)
    ]
    # t, flux, flux_err = binning_lc(folded_lc)
    binned_lc = folded_lc.bin(bins=500).remove_nans()
    t = binned_lc.time.value
    flux = binned_lc.flux.value
    flux_err = binned_lc.flux_err.value

    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(
        folded_lc.time.value,
        folded_lc.flux.value,
        color="black",
        marker=".",
    )
    ax.errorbar(
        binned_time,
        binned_flux,
        binned_flux_err,
        color="red",
        marker=".",
    )
    plt.show()
    pdb.set_trace()
    """

    """
    # chisquareの移動平均を見るときに使う
    names = ["q1", "q2", "t0", "porb", "rp_rs", "a_rs",
                 "b", "norm", "theta", "phi", "tau", "r_in",
                 "r_out", "norm2", "norm3", "ecosw", "esinw"]
    p_df=pd.read_csv('/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/TOI495.01_128_959.csv')
    p_df.index = names
    pdic = p_df.output_value.to_dict()
    ring_flux_model = ring_model(t, pdic, mcmc_pvalues=None)
    each_chi_square = ((flux-ring_flux_model)/flux_err)**2
    fig = plt.figure()
    ax_lc = fig.add_subplot(2,1,1) #for plotting transit model and data
    ax_re = fig.add_subplot(2,1,2) #for plotting residuals
    binned_lc.errorbar(ax=ax_lc)
    ax_lc.plot(t, ring_flux_model, label='Model w/ ring', color='blue')
    residuals_ring = binned_lc - ring_flux_model
    #residuals_ring.plot(ax=ax_re, color='blue', alpha=0.3,  marker='.', zorder=1)
    #ax_re.plot(t, np.zeros(len(t)), color='black', zorder=2)
    y = np.convolve(each_chi_square,np.ones(70)/70, mode='same')
    ax_lc.legend()
    ax_re.plot(t,y)
    plt.tight_layout()
    plt.show()
    """
    # t = np.linspace(-0.2, 0.2, 300)
    ###ring model fitting by minimizing chi_square###
    best_res_dict = {}
    for n in range(30):
        noringnames = ["t0", "per", "rp", "a", "b", "ecc", "w", "q1", "q2"]
        # values = [0.0, 4.0, 0.08, 8.0, 83.0, 0.0, 90.0, 0.2, 0.2]
        # noringvalues = [0, period, rp_rs, a_rs, 83.0, 0.0, 90.0, 0.2, 0.2]
        if np.isnan(rp_rs):
            rp_rs = rp_rs
        else:
            rp_rs = np.random.uniform(0.05, 0.1)

        noringvalues = [
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
        noringmins = [-0.2, period * 0.8, 0.01, 1, 0, 0, 90, 0.0, 0.0]
        noringmaxes = [0.2, period * 1.2, 0.5, 100, 1.0, 0.8, 90, 1.0, 1.0]
        noringvary_flags = [
            True,
            False,
            True,
            True,
            True,
            False,
            False,
            True,
            True,
        ]
        no_ring_params = set_params_lm(
            noringnames,
            noringvalues,
            noringmins,
            noringmaxes,
            noringvary_flags,
        )
        no_ring_res = lmfit.minimize(
            no_ring_transitfit,
            no_ring_params,
            args=(np.array(t), flux, flux_err, noringnames),
            max_nfev=1000,
        )
        if no_ring_res.params["t0"].stderr != None:
            if (
                np.isfinite(no_ring_res.params["t0"].stderr)
                and no_ring_res.redchi < 10
            ):
                red_redchi = no_ring_res.redchi - 1
                best_res_dict[red_redchi] = no_ring_res
    no_ring_res = sorted(best_res_dict.items())[0][1]

    best_ring_res_dict = {}
    for m in range(1):
        names = [
            "q1",
            "q2",
            "t0",
            "porb",
            "rp_rs",
            "a_rs",
            "b",
            "norm",
            "theta",
            "phi",
            "tau",
            "r_in",
            "r_out",
            "norm2",
            "norm3",
            "ecosw",
            "esinw",
        ]
        values = [
            no_ring_res.params.valuesdict()["q1"],
            no_ring_res.params.valuesdict()["q2"],
            no_ring_res.params.valuesdict()["t0"],
            period,
            no_ring_res.params.valuesdict()["rp"],
            no_ring_res.params.valuesdict()["a"],
            no_ring_res.params.valuesdict()["b"],
            1,
            np.random.uniform(1e-5, np.pi - 1e-5),
            np.random.uniform(0.0, np.pi),
            1,
            np.random.uniform(1.01, 2.44),
            np.random.uniform(1.02, 2.44),
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        saturnlike_values = [
            0.26,
            0.36,
            0.0,
            period,
            0.123,
            3.81,
            0.00,
            1,
            45 * np.pi / 180,
            0 * np.pi / 180,
            1,
            1.53,
            1.95,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        mins = [
            0.0,
            0.0,
            -0.1,
            0.0,
            0.01,
            1.0,
            0.0,
            0.9,
            0.0,
            -np.pi / 2,
            -np.pi / 2,
            1.00,
            1.01,
            -0.1,
            -0.1,
            0.0,
            0.0,
        ]

        maxes = [
            1.0,
            1.0,
            0.1,
            100.0,
            0.5,
            100.0,
            1.0,
            1.1,
            np.pi / 2,
            np.pi / 2,
            1.0,
            2.45,
            2.45,
            0.1,
            0.1,
            0.0,
            0.0,
        ]

        vary_flags = [
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            False,
            True,
            True,
            False,
            True,
            True,
            False,
            False,
            False,
            False,
        ]

        vary_flags = [  # t0だけずれてるかチェックしたいとき用
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
        params = set_params_lm(
            names, saturnlike_values, mins, maxes, vary_flags
        )
        params_df = pd.DataFrame(
            list(zip(values, saturnlike_values, mins, maxes)),
            columns=["values", "saturnlike_values", "mins", "maxes"],
            index=names,
        )
        vary_dic = dict(zip(names, vary_flags))
        params_df = params_df.join(
            pd.DataFrame.from_dict(
                vary_dic, orient="index", columns=["vary_flags"]
            )
        )
        df_for_mcmc = params_df[params_df["vary_flags"] == True]

        pdic = params_df["saturnlike_values"].to_dict()
        try:
            ring_res = lmfit.minimize(
                ring_transitfit,
                params,
                args=(t, flux, flux_err, names),
                max_nfev=100,
            )
        except ValueError:
            print("Value Error")
            print(m, TOInumber)
            print(params_df["values"])
            continue

        if ring_res.params["t0"].stderr != None:
            if np.isfinite(ring_res.params["t0"].stderr):
                red_redchi = ring_res.redchi - 1
                best_ring_res_dict[red_redchi] = ring_res
                """
                F_obs = (
                    (no_ring_res.chisqr - ring_res.chisqr)
                    / (ring_res.nvarys - no_ring_res.nvarys)
                ) / (ring_res.chisqr / (ring_res.ndata - ring_res.nvarys - 1))
                if F_obs > 0:
                    p_value = (
                        1
                        - integrate.quad(
                            lambda x: scipy.stats.f.pdf(
                                x,
                                ring_res.ndata - ring_res.nvarys - 1,
                                ring_res.nvarys - no_ring_res.nvarys,
                            ),
                            0,
                            F_obs,
                        )[0]
                    )
                    input_df = pd.DataFrame.from_dict(
                        params.valuesdict(),
                        orient="index",
                        columns=["input_value"],
                    )
                    output_df = pd.DataFrame.from_dict(
                        ring_res.params.valuesdict(),
                        orient="index",
                        columns=["output_value"],
                    )
                    input_df = input_df.applymap(lambda x: "{:.6f}".format(x))
                    output_df = output_df.applymap(
                        lambda x: "{:.6f}".format(x)
                    )
                    result_df = input_df.join(
                        (
                            output_df,
                            pd.Series(
                                vary_flags, index=names, name="vary_flags"
                            ),
                        )
                    )
                    # os.makedirs(f'./lmfit_result/fit_p_data/{TOInumber}', exist_ok=True)
                    os.makedirs(
                        f"./simulation/fit_p_data/{TOInumber}", exist_ok=True
                    )
                    result_df.to_csv(
                        f"./simulation/fit_p_data/{TOInumber}/{TOInumber}_{ring_res.redchi:.2f}_{m}.csv",
                        header=True,
                        index=False,
                    )
                    plot_ring(
                        rp_rs=ring_res.params["rp_rs"].value,
                        rin_rp=ring_res.params["r_in"].value,
                        rout_rin=ring_res.params["r_out"].value,
                        b=ring_res.params["b"].value,
                        theta=ring_res.params["theta"].value,
                        phi=ring_res.params["phi"].value,
                        file_name=f"{TOInumber}_{ring_res.redchi:.2f}_{m}.png",
                    )
                    fig = plt.figure()
                    ax_lc = fig.add_subplot(
                        2, 1, 1
                    )  # for plotting transit model and data
                    ax_re = fig.add_subplot(2, 1, 2)  # for plotting residuals
                    # elapsed_time = time.time() - start
                    # print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
                    ring_flux_model = ring_transitfit(
                        ring_res.params,
                        t,
                        flux,
                        flux_err,
                        names,
                        return_model=True,
                    )
                    noring_flux_model = no_ring_transitfit(
                        no_ring_res.params,
                        t,
                        flux,
                        flux_err,
                        noringnames,
                        return_model=True,
                    )
                    binned_lc.errorbar(ax=ax_lc)
                    ax_lc.plot(
                        t, ring_flux_model, label="Model w/ ring", color="blue"
                    )
                    ax_lc.plot(
                        t,
                        noring_flux_model,
                        label="Model w/o ring",
                        color="red",
                    )
                    residuals_ring = binned_lc - ring_flux_model
                    residuals_no_ring = binned_lc - noring_flux_model
                    residuals_ring.plot(
                        ax=ax_re, color="blue", alpha=0.3, marker=".", zorder=1
                    )
                    residuals_no_ring.plot(
                        ax=ax_re, color="red", alpha=0.3, marker=".", zorder=1
                    )
                    ax_re.plot(t, np.zeros(len(t)), color="black", zorder=2)
                    ax_lc.legend()
                    ax_lc.set_title(
                        f"w/ chisq:{ring_res.chisqr:.0f}/{ring_res.nfree:.0f} w/o chisq:{no_ring_res.chisqr:.0f}/{no_ring_res.nfree:.0f}"
                    )
                    # ax_lc.set_title(f'w/ AIC:{ring_res.aic:.2f} w/o AIC:{no_ring_res.aic:.2f}')
                    plt.tight_layout()
                    # os.makedirs(f'./lmfit_result/transit_fit/{TOInumber}', exist_ok=True)
                    os.makedirs(
                        f"./simulation/transit_fit/{TOInumber}", exist_ok=True
                    )
                    plt.savefig(
                        f"./simulation/transit_fit/{TOInumber}/{TOInumber}_{p_value}_{m}.png",
                        header=False,
                        index=False,
                    )
                    # plt.show()
                    plt.close()
                """
    ring_res = sorted(best_ring_res_dict.items())[0][1]
    fig = plt.figure()
    ax_lc = fig.add_subplot(2, 1, 1)  # for plotting transit model and data
    ax_re = fig.add_subplot(2, 1, 2)  # for plotting residuals
    # elapsed_time = time.time() - start
    # print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    ring_flux_model = ring_transitfit(
        ring_res.params,
        t,
        flux,
        flux_err,
        names,
        return_model=True,
    )
    noring_flux_model = no_ring_transitfit(
        no_ring_res.params,
        t,
        flux,
        flux_err,
        noringnames,
        return_model=True,
    )
    ax_lc.errorbar(
        t,
        flux,
        flux_err,
        color="black",
        marker=".",
        linestyle="None",
    )
    ax_lc.plot(t, ring_flux_model, label="Model w/ ring", color="blue")
    ax_lc.plot(t, noring_flux_model, label="Model w/o ring", color="red")
    residuals_ring = flux - ring_flux_model
    residuals_no_ring = flux - noring_flux_model
    ax_re.plot(
        t, residuals_ring, color="blue", alpha=0.3, marker=".", zorder=1
    )
    ax_re.plot(
        t, residuals_no_ring, color="red", alpha=0.3, marker=".", zorder=1
    )
    ax_re.plot(t, np.zeros(len(t)), color="black", zorder=2)
    ax_lc.legend()
    ax_lc.set_title(
        f"w/ chisq:{np.sum(((flux-ring_flux_model)/flux_err)**2):.0f}/{len(flux)} w/o chisq:{no_ring_res.chisqr:.0f}/{no_ring_res.nfree:.0f}"
    )
    # ax_lc.set_title(f'w/ AIC:{ring_res.aic:.2f} w/o AIC:{no_ring_res.aic:.2f}')
    plt.tight_layout()
    plt.show()
    # plt.close()
    sys.exit()
    F_obs = (
        (no_ring_res.chisqr - ring_res.chisqr)
        / (ring_res.nvarys - no_ring_res.nvarys)
    ) / (ring_res.chisqr / (ring_res.ndata - ring_res.nvarys - 1))
    if F_obs > 0:
        p_value = (
            1
            - integrate.quad(
                lambda x: scipy.stats.f.pdf(
                    x,
                    ring_res.ndata - ring_res.nvarys - 1,
                    ring_res.nvarys - no_ring_res.nvarys,
                ),
                0,
                F_obs,
            )[0]
        )
    else:
        p_value = "None"
    with open(f"./simulation/fit_report/{TOInumber}.txt", "a") as f:
        print("no ring transit fit report:\n", file=f)
        print(lmfit.fit_report(no_ring_res), file=f)
        print("ring transit fit report:\n", file=f)
        print(lmfit.fit_report(ring_res), file=f)
        print(f"F_obs: {F_obs}", file=f)
        print(f"p_value: {p_value}", file=f)

    # if abs(ring_res.redchi-1) < abs(no_ring_res.redchi-1):
    # with open(f'./lmfit_result/ring_hypo.txt', 'a') as f:
    # print(TOInumber+ '\n', file = f)
    # fit_report = lmfit.fit_report(ring_res)
    # print(fit_report)

sys.exit()
