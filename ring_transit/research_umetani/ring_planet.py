# -*- coding: utf-8 -*-
import os
import pdb
import sys
import warnings
from multiprocessing import Pool, cpu_count

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
from scipy.optimize import root
from tqdm import tqdm

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
    # model_flux = np.nan_to_num(model_flux)
    return model_flux


# リングありモデルをfitting
def ring_transitfit(params, x, data, eps_data, p_names, return_model=False):
    # start =time.time()
    model = ring_model(x, params.valuesdict())
    chi_square = np.sum(((data - model) / eps_data) ** 2)
    # print(params)
    # print(chi_square)
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


def make_simulation_data(
    period, b, rp_rs, bin_error, theta=45, phi=0, r_in=1.01, r_out=1.70
):
    """make_simulation_data"""
    t = np.linspace(-0.08, 0.08, 500)
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

    saturnlike_values = [
        0.26,
        0.36,
        0,
        period,
        rp_rs,
        3.81,
        b,
        1,
        theta * np.pi / 180,
        phi * np.pi / 180,
        1,
        r_in,
        r_out,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    # --土星likeなTOI495.01のパラメータで作成したモデル--#
    pdic_saturnlike = dict(zip(names, saturnlike_values))

    ymodel = (
        ring_model(t, pdic_saturnlike)
        # + np.random.randn(len(t)) * 0.00001
        # * (1 + (np.sin((t / 0.6 + 1.2 * np.random.rand()) * np.pi) * 0.01))
    )
    """
    ymodel = (
        ring_model(t, pdic_saturnlike)
        + np.random.randn(len(t)) * 0.001
        + np.sin((t / 0.6 + 1.2 * np.random.rand()) * np.pi) * 0.01
    )
    """
    ymodel = ymodel  # * 15000
    yerr = np.array(t / t) * bin_error  # * 15000
    each_lc = lk.LightCurve(t, ymodel, yerr)

    # each_lc.time = each_lc.time + mid_transit_time + np.random.randn() * 0.01

    return each_lc


def no_ring_transit_model(t):
    p_names = ["t0", "per", "rp", "a", "b", "ecc", "w", "q1", "q2"]
    values = [
        0,
        period,
        0.3,
        3.81,
        0.00,
        0,
        90.0,
        0.26,
        0.36,
    ]
    maxes = [0.5, period * 1.2, 0.5, 1000, 1 + rp_rs, 0.8, 90, 1.0, 1.0]
    mins = [-0.5, period * 0.8, 0.01, 1, 0, 0, 90, 0.0, 0.0]
    vary_flags = [True, False, True, True, True, False, False, True, True]
    params = set_params_lm(p_names, values, mins, maxes, vary_flags)
    params_batman = set_params_batman(params, p_names)
    m = batman.TransitModel(params_batman, t)  # initializes model
    model = m.light_curve(params_batman)  # calculates light curve

    return model


def calc_depth(rp_rs, b, period, theta, phi):
    """calc_rp_rs"""
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
    saturnlike_values = [
        0.26,
        0.36,
        0,
        period,
        rp_rs,
        3.81,
        b, 
        1,
        theta * np.pi / 180,
        phi * np.pi / 180,
        1,
        1.01,
        1.70,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    pdic = dict(zip(names, saturnlike_values))
    return ring_model([0], pdic)[0]


def residual_depth(param, b, period, theta, phi, depth):
    rp_rs = param["rp_rs"].value
    return calc_depth(rp_rs, b, period, theta, phi) - depth


def get_rp_rs(depth: float, b:float, period:float, theta: float, phi: float) -> float:
    param = lmfit.Parameters()
    param.add(
        "rp_rs",
        value=0.1,
        min=0.001,
        max=0.5,
        vary=True,
    )
    res = lmfit.minimize(
        residual_depth,
        param,
        args=(
            b,
            period,
            theta,
            phi,
            depth,
        ),
        max_nfev=1000,
    )

    return res.params["rp_rs"].value


def noring_params_setting(period, rp_rs):
    noringnames = [
                    "t0",
                    "per",
                    "rp",
                    "a",
                    "b",
                    "ecc",
                    "w",
                    "q1",
                    "q2",
    ]
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
    noringmins = [
        -0.2,
        period * 0.8,
        0.01,
        1,
        0,
        0,
        90,
        0.0,
        0.0,
    ]
    noringmaxes = [
        0.2,
        period * 1.2,
        0.5,
        100,
        1.0,
        0.8,
        90,
        1.0,
        1.0,
    ]
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
    
    return no_ring_params


def ring_params_setting(no_ring_res, b, period, rp_rs, theta, phi):
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
        rp_rs,
        3.81,
        b,
        1,
        theta * np.pi / 180,
        phi * np.pi / 180,
        1,
        1.01,
        1.70,
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
        True,
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
    params = set_params_lm(
        names, saturnlike_values, mins, maxes, vary_flags
    )

    return params


def process_bin_error(bin_error, b, rp_rs, theta, phi, period, min_flux):
    print(b, theta, phi, min_flux, bin_error)
    binned_lc = make_simulation_data(
        period, b, rp_rs, bin_error, theta=theta, phi=phi
    )
    # binned_lc = folded_lc.bin(bins=500).remove_nans()
    t = binned_lc.time.value
    flux = binned_lc.flux.value
    flux_err = binned_lc.flux_err.value

    # t = np.linspace(-0.2, 0.2, 300)
    ###no ring model fitting by minimizing chi_square###
    best_res_dict = {}
    for n in range(30):
        no_ring_params = noring_params_setting(period, rp_rs)
        no_ring_res = lmfit.minimize(
            no_ring_transitfit,
            no_ring_params,
            args=(np.array(t), flux, flux_err, list(no_ring_params.keys())),
            max_nfev=1000,
        )
        red_redchi = no_ring_res.redchi - 1
        best_res_dict[red_redchi] = no_ring_res
    no_ring_res = sorted(best_res_dict.items())[0][1]

    ###ring model fitting by minimizing chi_square###
    best_ring_res_dict = {}
    for _ in range(1):
        params = ring_params_setting(no_ring_res, b, period, rp_rs, theta, phi)
        try:
            ring_res = lmfit.minimize(
                ring_transitfit,
                params,
                args=(t, flux, flux_err, list(params.keys())),
                max_nfev=100,
            )
        except ValueError:
            print("Value Error")
            print(list(params.values()))
            continue
        red_redchi = ring_res.redchi - 1
        best_ring_res_dict[red_redchi] = ring_res
    ring_res = sorted(best_ring_res_dict.items())[0][1]
    fig = plt.figure()
    ax_lc = fig.add_subplot(
        2, 1, 1
    )  # for plotting transit model and data
    ax_re = fig.add_subplot(2, 1, 2)  # for plotting residuals
    ring_flux_model = ring_transitfit(
        ring_res.params,
        t,
        flux,
        flux_err,
        list(params.keys()),
        return_model=True,
    )
    noring_flux_model = no_ring_transitfit(
        no_ring_res.params,
        t,
        flux,
        flux_err,
        list(no_ring_params.keys()),
        return_model=True,
    )
    ax_lc.errorbar(
        t,
        flux,
        flux_err,
        color="black",
        marker=".",
        linestyle="None",
        zorder=2,
    )
    ax_lc.plot(
        t, ring_flux_model, label="Model w/ ring", color="blue"
    )
    ax_lc.plot(
        t, noring_flux_model, label="Model w/o ring", color="red"
    )
    residuals_ring = flux - ring_flux_model
    residuals_no_ring = flux - noring_flux_model
    ax_re.plot(
        t,
        residuals_ring,
        color="blue",
        alpha=0.3,
        marker=".",
        zorder=1,
    )
    ax_re.plot(
        t,
        residuals_no_ring,
        color="red",
        alpha=0.3,
        marker=".",
        zorder=1,
    )
    ax_re.plot(t, np.zeros(len(t)), color="black", zorder=2)
    ax_lc.legend()
    ax_lc.set_title(
        f"w/o chisq:{no_ring_res.chisqr:.0f}/{no_ring_res.nfree:.0f}"
    )
    plt.tight_layout()
    os.makedirs(
        f"./depth_error/figure/b_{b}/{theta}deg_{phi}deg",
        exist_ok=True,
    )
    plt.savefig(
        f"./depth_error/figure/b_{b}/{theta}deg_{phi}deg/{min_flux}_{bin_error}.png"
    )
    plt.close()
    ring_model_chisq = ring_res.ndata
    F_obs = (
        (no_ring_res.chisqr + ring_model_chisq - ring_model_chisq)
        / (ring_res.nvarys - no_ring_res.nvarys)
    ) / (ring_model_chisq / (ring_res.ndata - ring_res.nvarys - 1))
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
    os.makedirs(
        f"./depth_error/data/b_{b}/{theta}deg_{phi}deg",
        exist_ok=True,
    )
    with open(
        f"./depth_error/data/b_{b}/{theta}deg_{phi}deg/TOI495.01_{min_flux}_{bin_error}.txt",
        "w",
    ) as f:
        print("no ring transit fit report:\n", file=f)
        print(lmfit.fit_report(no_ring_res), file=f)
        print("ring transit fit report:\n", file=f)
        print(lmfit.fit_report(ring_res), file=f)
        print(f"F_obs: {F_obs}", file=f)
        print(f"p_value: {p_value}", file=f)


def process_bin_error_wrapper(args):
    return process_bin_error(*args)


def main():
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
        rp_rs=0.1,
        rin_rp=1.01,
        rout_rin=1.70,
        b=0.0,
        theta=45 * np.pi / 180,
        phi=15 * np.pi / 180,
        file_name="test.png",
    )
    sys.exit()
    """

    TOI = "495.01"
    print(TOI)
    TOInumber = "TOI" + TOI
    param_df = df[df["TOI"] == TOI]
    period = param_df["Period (days)"].values[0]

    b_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 0]
    min_flux_list = [
        0.999,
        0.998,
        0.997,
        0.996,
        0.995,
        0.994,
        0.993,
        0.992,
        0.991,
        0.99,
        0.98,
        0.97,
        0.96,
        0.95,
        0.94,
        0.93,
        0.92,
        0.91,
        0.90,
    ]
    theta_list = [45, 3, 15, 30]
    phi_list = [45, 0, 15, 30]
    for b in b_list:
        for theta in theta_list:
            for phi in phi_list:
                for min_flux in min_flux_list:
                    bin_error_list = np.arange(0.0001, 0.0041, 0.0001)
                    bin_error_list = np.around(bin_error_list, decimals=4)
                    new_bin_error_list = []
                    for bin_error in bin_error_list:
                        print(bin_error)
                        if os.path.isfile(f"./depth_error/figure/b_{b}/{theta}deg_{phi}deg/{min_flux}_{bin_error}.png"):
                            print(f"b_{b}/{theta}deg_{phi}deg/{min_flux}_{bin_error}.png is exist.")
                            continue     
                        else:
                            new_bin_error_list.append(bin_error)

                    if len(new_bin_error_list) == 0:
                        continue

                    # get rp_rs value from transit depth
                    rp_rs = get_rp_rs(min_flux, b, period, theta, phi)

                    src_datas = list(
                                map(
                                    lambda x: [
                                        x,
                                        b,
                                        rp_rs,
                                        theta,
                                        phi,
                                        period,
                                        min_flux
                                    ],
                                    new_bin_error_list,
                                ))
                    with Pool(cpu_count() - 1) as p:
                        p.map(process_bin_error_wrapper, src_datas)


if __name__ == '__main__':
    main()
