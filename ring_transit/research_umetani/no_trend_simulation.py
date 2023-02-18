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


class PlotLightcurve:
    def __init__(self, savedir, savefile, lc):
        self.savedir = savedir
        self.savefile = savefile
        self.lc = lc

    def plot_lightcurve(self):
        self.lc.errorbar()

    def save(self):
        os.makedirs(self.savedir, exist_ok=True)
        plt.savefig(f"{self.savedir}/{self.savefile}")
        plt.close()


class PlotLightcurveWithModel(PlotLightcurve):
    def __init__(self, savedir, savefile, lc, model: dict, outliers):
        super().__init__(savedir, savefile, lc)
        self.t = self.lc.time.value
        self.y = self.lc.flux.value
        self.yerr = self.lc.flux_err.value
        self.model = model
        self.outliers = outliers
        fig = plt.figure()
        self.ax_lc = fig.add_subplot(2, 1, 1)
        self.ax_re = fig.add_subplot(2, 1, 2)  # for plotting residuals
        self.residuals = self.lc - self.model["fitting model"][0]

    def plot_lightcurve(
        self, label="data", color="gray", marker=".", alpha=0.3
    ):
        self.lc.errorbar(
            ax=self.ax_lc, label=label, color=color, marker=marker, alpha=alpha
        )

    def plot_model(self):
        for label, (model, color) in self.model.items():
            self.ax_lc.plot(self.t, model, label=label, color=color)

    def plot_residuals(
        self,
        res_color="gray",
        ref_line_color="black",
        alpha=0.3,
        marker=".",
        zorder=1,
    ):
        self.residuals.plot(
            ax=self.ax_re,
            color=res_color,
            alpha=alpha,
            marker=marker,
            zorder=1,
        )
        self.ax_re.plot(
            self.t, np.zeros(len(self.t)), color=ref_line_color, zorder=2
        )
        self.ax_re.set_ylabel("residuals")

    def plot_outliers(self, label="outliers", color="cyan", marker="."):
        try:
            outliers = vstack(self.outliers)
        except ValueError:
            pass
        else:
            outliers.errorbar(
                ax=self.ax_lc, color=color, label=label, marker=marker
            )

    def configs(self):
        chi_square = np.sum(
            ((self.y - self.model["fitting model"][0]) / self.yerr) ** 2
        )
        self.ax_lc.legend()
        self.ax_lc.set_title(
            f"chi square/dof: {int(chi_square)}/{len(self.y)} "
        )
        plt.tight_layout()


class PlotCurvefit(PlotLightcurve):
    def __init__(self, savedir, savefile, lc, fit_res):
        super().__init__(savedir, savefile, lc)
        self.fit_res = fit_res

    def plot(self):
        self.fit_res.plot()


def calc_data_survival_rate(lc, duration):
    data_n = len(lc.flux)
    max_data_n = duration * 5 / (2 / 60 / 24)
    data_survival_rate = (data_n / max_data_n) * 100
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
def no_ring_transitfit(params, lc, p_names, return_model=False):
    t = lc.time.value
    flux = lc.flux.value
    flux_err = lc.flux_err.value
    params_batman = set_params_batman(params, p_names)
    m = batman.TransitModel(params_batman, t)  # initializes model
    model = m.light_curve(params_batman)  # calculates light curve
    # chi_square = np.sum(((data - model) / eps_data) ** 2)
    # print(params)
    # print(chi_square)
    # visualize_plot_process(model, x, data, eps_data)
    if return_model:
        return model
    else:
        return (flux - model) / flux_err


def no_ring_transit_and_polynomialfit(params, lc, p_names, return_model=False):
    t = lc.time.value
    flux = lc.flux.value
    flux_err = lc.flux_err.value
    params_batman = set_params_batman(params, p_names)
    m = batman.TransitModel(params_batman, t)  # initializes model
    transit_model = m.light_curve(params_batman)  # calculates light curve
    poly_params = params.valuesdict()
    poly_model = np.polynomial.Polynomial(
        [
            poly_params["c0"],
            poly_params["c1"],
            poly_params["c2"],
            poly_params["c3"],
            poly_params["c4"],
        ]
    )
    polynomialmodel = poly_model(t)
    model = transit_model + polynomialmodel
    # chi_square = np.sum(((flux - model) / flux_err) ** 2)
    # print(params)
    # print(chi_square)
    # visualize_plot_process(model, x, data, eps_data)
    if return_model:
        return model, transit_model, polynomialmodel
    else:
        return (flux - model) / flux_err


def visualize_plot_process(model, x, data, eps_data):
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)  # for plotting residuals
    try:
        ax1.errorbar(x=x, y=data, yerr=eps_data, fmt=".k")
    except TypeError:
        ax1.scatter(x=x, y=data, color="r")
    ax1.plot(x, model, label="fitted line")
    ax1.set_ylabel("flux")
    residuals = data - model
    try:
        ax2.errorbar(x=x, y=residuals, yerr=eps_data, fmt=".k")
    except TypeError:
        ax2.scatter(x=x, y=residuals, color="r")
    ax2.plot(x, np.zeros(len(x)), color="red")
    ax2.set_xlabel("mid transit time[BJD] - 2457000")
    ax2.set_ylabel("residuals")
    plt.title(np.sum(((data - model) / eps_data) ** 2))
    plt.tight_layout()
    plt.pause(0.01)
    plt.close()


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
        f"{HOMEDIR}/fitting_result/data/calc_obs_transit_time/{TOInumber}.csv"
    )
    plt.errorbar(x=x, y=y, yerr=yerr, fmt=".k")
    plt.xlabel("mid transit time[BJD] - 2457000")
    plt.ylabel("O-C(hrs)")
    plt.tight_layout()
    os.makedirs(
        f"{HOMEDIR}/fitting_result/figure/simulation_TOI495.01/plussin_plusmodel/calc_obs_transit_time/",
        exist_ok=True,
    )
    plt.savefig(
        f"{HOMEDIR}/fitting_result/figure/simulation_TOI495.01/plussin_plusmodel/calc_obs_transit_time/{TOInumber}.png"
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
            f"{HOMEDIR}/fitting_result/figure/simulation_TOI495.01/plussin_plusmodel/estimate_period/",
            exist_ok=True,
        )
        plt.savefig(
            f"{HOMEDIR}/fitting_result/figure/simulation_TOI495.01/plussin_plusmodel/estimate_period/{TOInumber}.png"
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
    curvefit_params=None,
):
    """transit fitting"""
    flag_time = np.abs(lc.time.value) < 1.0
    lc = lc[flag_time]
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
                        # t0の初期値だけ動かし、function evalsの少なくてエラーが計算できないことを回避(no use)
                        pass
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
                args=(lc, p_names),
                max_nfev=10000,
            )
            # if res.params["t0"].stderr != None:
            # if np.isfinite(res.params["t0"].stderr):
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


def clip_outliers(
    lc,
    mask,
):
    print("outliers exist")
    lc = lc[~mask]

    return lc


def detect_outliers(lc, model):
    """
    if transit_and_poly_fit == True:
        flux_model, transit_model, polynomial_model = no_ring_transit_and_polynomialfit(
            res.params, lc, p_names, return_model=True
            )
    else:
        flux_model = no_ring_transitfit(
            res.params, lc, p_names, return_model=True
        )
    """
    residual_lc = lc.copy()
    residual_lc.flux = np.sqrt(np.square(lc.flux - model))
    _, mask = residual_lc.remove_outliers(sigma=5.0, return_mask=True)
    # inverse_mask = np.logical_not(mask)

    return mask


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
    poly_params = model.make_params(c0=1, c1=0, c2=0, c3=0, c4=0)
    result = model.fit(
        out_transit.flux.value, poly_params, x=out_transit.time.value
    )

    return result


def curvefit_normalize(each_lc, poly_params):
    poly_params = poly_params.valuesdict()
    poly_model = np.polynomial.Polynomial(
        [
            poly_params["c0"],
            poly_params["c1"],
            poly_params["c2"],
            poly_params["c3"],
            poly_params["c4"],
        ]
    )
    poly_model = poly_model(each_lc.time.value)

    # normalization
    each_lc.flux = each_lc.flux - poly_model
    each_lc.flux_err = each_lc.flux_err - poly_model

    return each_lc


def folding_lc_from_csv(loaddir):
    each_lc_list = []
    try:
        each_lc_list = []
        total_lc_csv = os.listdir(f"{loaddir}/")
        total_lc_csv = [i for i in total_lc_csv if "csv" in i]
        for each_lc_csv in total_lc_csv:
            each_table = ascii.read(f"{loaddir}/{each_lc_csv}")
            each_lc = lk.LightCurve(data=each_table)
            each_lc_list.append(each_lc)
    except ValueError:
        pass
    folded_lc = vstack(each_lc_list)
    folded_lc.sort("time")

    return folded_lc


def make_simulation_data(mid_transit_time):
    """make_simulation_data"""
    t = np.linspace(
        np.random.uniform(-0.31, -0.29), np.random.uniform(0.29, 0.31), 500
    )
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

    # --土星likeなTOI495.01のパラメータで作成したモデル--#
    pdic_saturnlike = dict(zip(names, saturnlike_values))
    ymodel = (
        ring_model(t, pdic_saturnlike)
        * (1 + (np.sin((t / 0.6 + 1.2 * np.random.rand()) * np.pi) * 0.01))
        + np.random.randn(len(t)) * 0.001
    )

    ymodel = (
        ring_model(t, pdic_saturnlike)
        + np.random.randn(len(t)) * 0.001
        + np.sin((t / 0.6 + 1.2 * np.random.rand()) * np.pi) * 0.01
    )

    ymodel = ymodel * 15000
    yerr = np.array(t / t) * 1e-3 * 15000
    each_lc = lk.LightCurve(t, ymodel, yerr)

    each_lc.time = each_lc.time + mid_transit_time + np.random.randn() * 0.01

    return each_lc


def ring_model(t, pdic, mcmc_pvalues=None):
    """
    if mcmc_pvalues is None:
        pass
    else:
        for i, param in enumerate(mcmc_params):
            # print(i, v[i])
            pdic[param] = mcmc_pvalues[i]
    """
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
    # cosi = b / a_rs
    # u = [2 * np.sqrt(q1) * q2, np.sqrt(q1) * (1 - 2 * q2)]
    # u1, u2 = u[0], u[1]
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


def ringfit(i, lc):
    # binned_lc = lc.bin(bins=500).normalize().remove_nans()
    binned_lc = lc
    t = binned_lc.time.value
    flux_data = binned_lc.flux.value
    flux_err_data = binned_lc.flux_err.value

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

    pdic = dict(zip(names, saturnlike_values))

    fig = plt.figure()
    ax_lc = fig.add_subplot(2, 1, 1)  # for plotting transit model and data
    ax_re = fig.add_subplot(2, 1, 2)  # for plotting residuals
    ring_flux_model = ring_model(t, pdic, mcmc_pvalues=None)
    # binned_lc.errorbar(ax=ax_lc)
    ax_lc.errorbar(t, flux_data, flux_err_data)
    ax_lc.plot(t, ring_flux_model, label="Model w/ ring", color="blue")
    residuals_ring = binned_lc - ring_flux_model
    residuals_ring.plot(
        ax=ax_re, color="blue", alpha=0.3, marker=".", zorder=1
    )
    ax_re.plot(t, np.zeros(len(t)), color="black", zorder=2)
    ax_re.set_ylim(-0.002, 0.002)
    ax_lc.legend()
    ax_lc.set_title(
        f"w/ chisq:{np.sum(((flux_data-ring_flux_model)/flux_err_data)**2):.0f}/{len(binned_lc)}"
    )
    # ax_lc.set_title(f'w/ AIC:{ring_res.aic:.2f} w/o AIC:{no_ring_res.aic:.2f}')
    plt.tight_layout()
    os.makedirs(
        f"{HOMEDIR}/fitting_result/figure/simulation_TOI495.01/plussin_plusmodel/45deg_0deg/badass/",
        exist_ok=True,
    )
    plt.savefig(
        f"{HOMEDIR}/fitting_result/figure/simulation_TOI495.01/plussin_plusmodel/45deg_0deg/badass/{i}.png"
    )
    plt.show()
    plt.close()
    pdb.set_trace()


"""定数の定義"""
HOMEDIR = (
    "/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani"
)
SAVE_BEFOREPROCESS_LC_DIR = f"{HOMEDIR}/fitting_result/figure/simulation_TOI495.01/plussin_plusmodel/45deg_0deg/before_process"
SAVE_BEFOREPROCESS_LC_DATA_DIR = f"{HOMEDIR}/fitting_result/data/simulation_TOI495.01/plussin_plusmodel/45deg_0deg/before_process"
EPOCH_NUM = 54

oridf = pd.read_csv(f"{HOMEDIR}/exofop_tess_tois.csv")
df = oridf[oridf["Planet SNR"] > 100]
df = df.sort_values("Planet SNR", ascending=False)
df["TOI"] = df["TOI"].astype(str)
TOIlist = df["TOI"]

TOI = 495.01
print("analysing: ", "TOI" + str(TOI))
TOI = str(TOI)

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

""""保存場所のセッティング"""
SAVE_CURVEFIT_DIR = f"{HOMEDIR}/fitting_result/figure/simulation_TOI495.01/plussin_plusmodel/45deg_0deg/curvefit/{TOInumber}"
SAVE_TRANSITFIT_DIR = f"{HOMEDIR}/fitting_result/figure/simulation_TOI495.01/plussin_plusmodel/45deg_0deg/each_lc/transit_fit/{TOInumber}"
SAVE_TRANSIT_POLYFIT_DIR = f"{HOMEDIR}/fitting_result/figure/simulation_TOI495.01/plussin_plusmodel/45deg_0deg/each_lc/transit_and_polyfit/{TOInumber}"
SAVE_1STPROCESS_LC_DIR = f"{HOMEDIR}/fitting_result/figure/simulation_TOI495.01/plussin_plusmodel/45deg_0deg/each_lc/after_curvefit/{TOInumber}"
SAVE_1STPROCESS_LC_DATA_DIR = f"{HOMEDIR}/fitting_result/data/simulation_TOI495.01/plussin_plusmodel/45deg_0deg/each_lc/calc_t0/{TOInumber}"
SAVE_1STMODELFIT_DIR = f"{HOMEDIR}/fitting_result/data/simulation_TOI495.01/plussin_plusmodel/45deg_0deg/each_lc/modelresult/1stloop/{TOInumber}"
SAVE_1STFOLDMODELFIT_DIR = f"{HOMEDIR}/fitting_result/data/simulation_TOI495.01/plussin_plusmodel/45deg_0deg/folded_lc/modelresult/1stloop/{TOInumber}"
SAVE_1STFOLD_LC_DIR = "/Users/u_tsubasa/Dropbox/ring_planet_research/folded_lc/figure/simulation_TOI495.01/plussin_plusmodel/45deg_0deg/calc_t0"
SAVE_1STFOLD_LC_DATA_DIR = "/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/fitting_result/data/simulation_TOI495.01/plussin_plusmodel/45deg_0deg/folded_lc/calc_t0"
SAVE_2NDMODELFIT_DIR = f"{HOMEDIR}/fitting_result/data/simulation_TOI495.01/plussin_plusmodel/45deg_0deg/each_lc/modelresult/2ndloop/{TOInumber}"
SAVE_2NDPROCESS_LC_DIR = f"{HOMEDIR}/fitting_result/figure/simulation_TOI495.01/plussin_plusmodel/45deg_0deg/each_lc/after_curvefit/{TOInumber}"
SAVE_2NDPROCESS_LC_DATA_DIR = f"{HOMEDIR}/fitting_result/data/simulation_TOI495.01/plussin_plusmodel/45deg_0deg/each_lc/obs_t0/{TOInumber}"
SAVE_2NDFOLD_LC_DIR = "/Users/u_tsubasa/Dropbox/ring_planet_research/folded_lc/figure/simulation_TOI495.01/plussin_plusmodel/45deg_0deg/obs_t0"
SAVE_2NDFOLD_LC_DATA_DIR = "/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/fitting_result/data/simulation_TOI495.01/plussin_plusmodel/45deg_0deg/folded_lc/obs_t0"
SAVE_2NDFOLDMODELFIT_DIR = f"{HOMEDIR}/fitting_result/data/simulation_TOI495.01/plussin_plusmodel/45deg_0deg/folded_lc/modelresult/2ndloop/{TOInumber}"


# folded_lc = folding_lc_from_csv(SAVE_BEFOREPROCESS_LC_DATA_DIR)
# folded_lc.write(f"{SAVE_2NDFOLD_LC_DATA_DIR}/{TOInumber}.csv", overwrite=True)
# pdb.set_trace()


"""各エポックで外れ値除去と多項式フィッティング"""
# 値を格納するリストの定義
outliers = []
t0list = []
t0errlist = []
num_list = np.arange(EPOCH_NUM)
transit_time_list = [transit_time + period * i for i in range(EPOCH_NUM)]
# ax = lc.scatter()

for i, mid_transit_time in enumerate(transit_time_list):
    print(f"preprocessing...epoch: {i}")

    each_lc = make_simulation_data(mid_transit_time)
    plot_lc = PlotLightcurve(
        savedir=SAVE_BEFOREPROCESS_LC_DIR, savefile=f"{i}.png", lc=each_lc
    )
    plot_lc.plot_lightcurve()
    plot_lc.save()
    os.makedirs(SAVE_BEFOREPROCESS_LC_DATA_DIR, exist_ok=True)
    each_lc.write(f"{SAVE_BEFOREPROCESS_LC_DATA_DIR}/{i}.csv", overwrite=True)
    # each_table = ascii.read(f"{HOMEDIR}/fitting_result/data/simulation_TOI495.01/plussin_plusmodel/before_process/{i}.csv")
    # each_lc = lk.LightCurve(data=each_table)

    each_lc = (
        each_lc.fold(period=period, epoch_time=mid_transit_time)
        .normalize()
        .remove_nans()
    )

    # each_lc = each_lc.remove_nans()
    """外れ値除去と多項式フィッティングを外れ値が検知されなくなるまで繰り返す"""
    while True:
        # curvefitを正確にするためtransitfitでt0を求めておく
        transit_res = transit_fitting(each_lc, rp_rs, period)
        curvefit_res = curve_fitting(each_lc, duration, transit_res)
        # visualize_plot_process(model, x, data, eps_data)
        plot_lc = PlotCurvefit(
            savedir=SAVE_CURVEFIT_DIR,
            savefile=f"{TOInumber}_{str(i)}.png",
            lc=each_lc,
            fit_res=curvefit_res,
        )
        plot_lc.plot()
        plot_lc.save()

        each_lc = curvefit_normalize(each_lc, curvefit_res.params)
        transit_res = transit_fitting(each_lc, rp_rs, period)
        transit_model = no_ring_transitfit(
            transit_res.params, each_lc, p_names, return_model=True
        )
        outlier_bools = detect_outliers(each_lc, transit_model)
        inverse_mask = np.logical_not(outlier_bools)
        if np.all(inverse_mask):
            plot_lc = PlotLightcurveWithModel(
                savedir=SAVE_TRANSITFIT_DIR,
                savefile=f"{TOInumber}_{str(i)}.png",
                lc=each_lc,
                model={"fitting model": (transit_model, "black")},
                outliers=outliers,
            )
            plot_lc.plot_lightcurve()
            plot_lc.plot_model()
            plot_lc.plot_residuals()
            plot_lc.plot_outliers()
            plot_lc.configs()
            plot_lc.save()

            each_lc.time = each_lc.time - transit_res.params["t0"].value
            plot_after1stloop_lc = PlotLightcurve(
                savedir=SAVE_1STPROCESS_LC_DIR,
                savefile=f"{TOInumber}_{str(i)}.png",
                lc=each_lc,
            )
            plot_after1stloop_lc.plot_lightcurve()
            plot_after1stloop_lc.save()
            os.makedirs(SAVE_1STPROCESS_LC_DATA_DIR, exist_ok=True)
            each_lc.write(
                f"{SAVE_1STPROCESS_LC_DATA_DIR}/{TOInumber}_{str(i)}.csv",
                overwrite=True,
            )
            # save_each_lc(each_lc, "calc_t0")
            # t0のズレを補正したmidtransittimelistを作るためt0list,t0errlistそれぞれappend
            t0list.append(transit_res.params["t0"].value + mid_transit_time)
            t0errlist.append(transit_res.params["t0"].stderr)

            outliers = []

            break
        else:
            outliers.append(each_lc[outlier_bools])
            each_lc = clip_outliers(
                each_lc,
                outlier_bools,
            )

    os.makedirs(SAVE_1STMODELFIT_DIR, exist_ok=True)
    with open(
        f"{SAVE_1STMODELFIT_DIR}/{TOInumber}_{str(i)}.txt",
        "a",
    ) as f:
        print(lmfit.fit_report(transit_res), file=f)

    # ringfit(i, each_lc)
# plt.show()
# pdb.set_trace()
"""folded_lcに対してtransit fitting & remove outliers. transit parametersを得る"""
print("folding and calculate duration...")
time.sleep(1)
folded_lc = folding_lc_from_csv(
    loaddir=SAVE_1STPROCESS_LC_DATA_DIR,
)

outliers = []
while True:
    fold_res = transit_fitting(
        folded_lc, rp_rs, period, fitting_model=no_ring_transitfit
    )
    transit_model = no_ring_transitfit(
        fold_res.params, folded_lc, p_names, return_model=True
    )
    outlier_bools = detect_outliers(folded_lc, transit_model)
    inverse_mask = np.logical_not(outlier_bools)
    if np.all(inverse_mask):
        plot_lc = PlotLightcurveWithModel(
            savedir=SAVE_1STFOLD_LC_DIR,
            savefile=f"{TOInumber}.png",
            lc=folded_lc,
            model={"fitting model": (transit_model, "black")},
            outliers=outliers,
        )
        plot_lc.plot_lightcurve()
        plot_lc.plot_model()
        plot_lc.plot_residuals()
        plot_lc.plot_outliers()
        plot_lc.configs()
        plot_lc.save()

        os.makedirs(SAVE_1STFOLD_LC_DATA_DIR, exist_ok=True)
        folded_lc.write(
            f"{SAVE_1STFOLD_LC_DATA_DIR}/{TOInumber}.csv", overwrite=True
        )
        os.makedirs(SAVE_1STFOLDMODELFIT_DIR, exist_ok=True)
        with open(
            f"{SAVE_1STFOLDMODELFIT_DIR}/{TOInumber}_folded.txt",
            "a",
        ) as f:
            print(lmfit.fit_report(fold_res), file=f)
        break
    else:
        outliers.append(folded_lc[outlier_bools])
        folded_lc = clip_outliers(
            folded_lc,
            outlier_bools,
        )

transit_time_list = np.array(transit_time_list)
# _, _ = calc_obs_transit_time(t0list, t0errlist, num_list, transit_time_list, transit_time_error)
"""fittingで得たtransit time listを反映"""
obs_t0_list = t0list
outliers = []
t0list = []
t0errlist = []
num_list = []

"""transit parametersをfixして、baseline,t0を決める"""
for i, mid_transit_time in enumerate(obs_t0_list):
    print(f"reprocessing...epoch: {i}")
    each_table = ascii.read(f"{SAVE_BEFOREPROCESS_LC_DATA_DIR}/{i}.csv")
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
                f"{HOMEDIR}/fitting_result/figure/simulation_TOI495.01/plussin_plusmodel/error_lc/under_90%_data/obs_t0/{TOInumber}",
                exist_ok=True,
            )
            plt.savefig(
                f"{HOMEDIR}/fitting_result/figure/simulation_TOI495.01/plussin_plusmodel/error_lc/under_90%_data/obs_t0/{TOInumber}/{TOInumber}_{str(i)}.png"
            )
            plt.close()
        t0list.append(mid_transit_time)
        t0errlist.append(np.nan)
        continue
    else:
        num_list.append(i)

    while True:
        # curvefitを正確に行うために1回transitfitしている
        transit_res = transit_fitting(each_lc, rp_rs, period)
        # 初期値を得るためのcurvefit
        curvefit_res = curve_fitting(each_lc, duration, transit_res)
        curvefit_res.params["c0"].value = curvefit_res.params["c0"] - 1

        res = transit_fitting(
            each_lc,
            rp_rs,
            period,
            fitting_model=no_ring_transit_and_polynomialfit,
            transitfit_params=fold_res.params,
            curvefit_params=curvefit_res.params,
        )
        (
            flux_model,
            transit_model,
            polynomial_model,
        ) = no_ring_transit_and_polynomialfit(
            res.params, each_lc, p_names, return_model=True
        )
        outlier_bools = detect_outliers(each_lc, flux_model)
        inverse_mask = np.logical_not(outlier_bools)
        if np.all(inverse_mask):
            # each_lc.time = each_lc.time - res.params["t0"].value
            plot_lc = PlotLightcurveWithModel(
                savedir=SAVE_TRANSIT_POLYFIT_DIR,
                savefile=f"{TOInumber}_{str(i)}.png",
                lc=each_lc,
                outliers=outliers,
                model={
                    "transit model": (transit_model, "blue"),
                    "polynomial model": (polynomial_model + 1, "red"),
                    "fitting model": (flux_model, "black"),
                },
            )
            plot_lc.plot_lightcurve()
            plot_lc.plot_model()
            plot_lc.plot_residuals()
            plot_lc.plot_outliers()
            plot_lc.configs()
            plot_lc.save()

            each_lc = curvefit_normalize(each_lc, res.params)
            each_lc.time = each_lc.time - res.params["t0"].value
            plot_after2ndloop_lc = PlotLightcurve(
                savedir=SAVE_2NDPROCESS_LC_DIR,
                savefile=f"{TOInumber}_{str(i)}.png",
                lc=each_lc,
            )
            plot_after2ndloop_lc.plot_lightcurve()
            plot_after2ndloop_lc.save()
            os.makedirs(SAVE_2NDPROCESS_LC_DATA_DIR, exist_ok=True)
            each_lc.write(
                f"{SAVE_2NDPROCESS_LC_DATA_DIR}/{TOInumber}_{str(i)}.csv",
                overwrite=True,
            )
            # save_each_lc(each_lc, "obs_t0")
            outliers = []
            break
        else:
            outliers.append(each_lc[outlier_bools])
            each_lc = clip_outliers(
                each_lc,
                outlier_bools,
            )
    os.makedirs(
        SAVE_2NDMODELFIT_DIR,
        exist_ok=True,
    )
    with open(
        f"{SAVE_2NDMODELFIT_DIR}/{TOInumber}_{str(i)}.txt",
        "a",
    ) as f:
        print(lmfit.fit_report(res), file=f)

    # ringfit(i, each_lc)

"""最終的なfolded_lcを生成する。"""
print("refolding...")
time.sleep(1)
folded_lc = folding_lc_from_csv(
    loaddir=SAVE_2NDPROCESS_LC_DATA_DIR,
)
outliers = []
while True:
    fold_res = transit_fitting(
        folded_lc, rp_rs, period, fitting_model=no_ring_transitfit
    )
    transit_model = no_ring_transitfit(
        fold_res.params, folded_lc, p_names, return_model=True
    )
    outlier_bools = detect_outliers(folded_lc, transit_model)
    inverse_mask = np.logical_not(outlier_bools)
    if np.all(inverse_mask):
        # save result
        plot_lc = PlotLightcurveWithModel(
            savedir=SAVE_2NDFOLD_LC_DIR,
            savefile=f"{TOInumber}.png",
            lc=folded_lc,
            model={"fitting model": (transit_model, "black")},
            outliers=outliers,
        )
        plot_lc.plot_lightcurve()
        plot_lc.plot_model()
        plot_lc.plot_residuals()
        plot_lc.plot_outliers()
        plot_lc.configs()
        plot_lc.save()

        os.makedirs(SAVE_2NDFOLD_LC_DATA_DIR, exist_ok=True)
        folded_lc.write(
            f"{SAVE_2NDFOLD_LC_DATA_DIR}/{TOInumber}.csv", overwrite=True
        )
        os.makedirs(
            SAVE_2NDFOLDMODELFIT_DIR,
            exist_ok=True,
        )
        with open(
            f"{SAVE_2NDFOLDMODELFIT_DIR}/{TOInumber}_folded.txt",
            "a",
        ) as f:
            print(lmfit.fit_report(fold_res), file=f)
        break
    else:
        outliers.append(folded_lc[outlier_bools])
        folded_lc = clip_outliers(
            folded_lc,
            outlier_bools,
        )
folded_lc.scatter()
plt.show()
print(f"Analysis completed: {TOInumber}")

# import pdb;pdb.set_trace()
