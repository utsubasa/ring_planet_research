# -*- coding: utf-8 -*-
import os
import pdb
import sys
import warnings
from concurrent import futures
from typing import Dict, List, Sequence, Tuple

import batman
import lightkurve as lk
import lmfit
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from astropy.io import ascii
from astropy.table import vstack
from numpy.linalg import svd
from scipy import integrate
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
        try:
            self.ax_lc.set_title(
                f"chi square/dof: {int(chi_square)}/{len(self.y)} "
            )
        except OverflowError:
            self.ax_lc.set_title(
                f"chi square/dof: {chi_square:.2e}/{len(self.y)} "
            )
        plt.tight_layout()


class PlotCurvefit(PlotLightcurve):
    def __init__(self, savedir, savefile, lc, fit_res):
        super().__init__(savedir, savefile, lc)
        self.fit_res = fit_res

    def plot(self):
        self.fit_res.plot()


def calc_b_and_a(
    period: float,
    period_err: float,
    ms: float,
    mp: float,
    rs: float,
    rp: float,
    duration: float,
) -> Tuple[float, float]:
    """Calculate impact parameter and orbital semimajor axis of the planet.
    Args:
        period (float): orbital period of the planet
        period_err (float): error of orbital period of the planet
        ms (float): stellar mass(kg)
        mp (float): planetary mass(kg)
        rs (float): stellar radius(m)
        rp (float): planetary radius(m)
        duration (float): transit duration(day)
    Returns:
        Tuple[float, float]: _description_
    """
    # calculate size of orbit using kepler's third law
    sec_period = period * 24 * 3600
    # sec_period_err = period_err * 24 * 3600
    g = 6.673e-11
    numerator = (sec_period**2) * (g * (ms + mp))
    denominator = 4 * (np.pi**2)

    a_m = (numerator / denominator) ** (1 / 3)
    # a_rs = a_m / 696340000  # unit solar radius
    # a_au = a_m * 6.68459e-12  # unit au
    A = (rs + rp) ** 2
    B = a_m**2
    C = np.sin(np.pi * duration / period) ** 2

    b = np.sqrt(np.abs(A - (B * C))) / rs

    return a_m, b


def calc_sigma(
    period: float, duration: float, toi: float, restrict_sector=True
):
    """Calculate sigma of each bin.
    Here, sigma mean variation of each bin of lightcurve.

    Args:
        param_df (pd.DataFrame): dataframe of the parameters of the planet
        period (float): orbital period of the planet
        duration (float): transit duration(day)
        lc (lk.LightCurve): lightcurve of the planet
    """
    # periodかdurationがnanの場合は計算しない
    if np.isnan(period) or np.isnan(duration):
        return 0
    # get lc
    while True:
        try:
            search_result = lk.search_lightcurve(
                f"TOI {str(toi)[:-3]}",
                mission="TESS",
                cadence="short",
                author="SPOC",
            )
            try:
                lc_collection = search_result.download_all()
            except AttributeError:
                lc_collection = search_result[0].download()
            break  # エラーが発生せずにループを抜ける
        except lk.utils.LightkurveError:
            continue  # エラーが発生した場合はループを継続する

    if lc_collection is None:
        with open("no_data_found_20230526.txt", "a") as f:
            print(f"{toi}", file=f)
        return None

    sigma_list = []
    # bins_list = []

    # lc_collectionからsectorが５６以上のものを除外してn_sectorを算出
    n_sector = len(lc_collection)
    if restrict_sector:
        for lc in lc_collection:
            if lc.sector >= 56:
                n_sector -= 1
    if n_sector == 0:
        return None
    observation_days = n_sector * 27.4
    n_transit = int(observation_days / period)
    for lc in lc_collection:
        if restrict_sector:
            if lc.sector >= 56:
                continue
        # sap fluxに変換
        # lc.flux = lc.sap_flux
        # lc.flux_err = lc.sap_flux_err

        # crowdsap等を考慮
        # lc = correct_sap_lc(lc)
        t_cdpp = 1.4 * duration / 500
        cdpp = lc.hdu[1].header["CDPP1_0"]
        sigma = (cdpp * np.sqrt(1 / t_cdpp) / np.sqrt(n_transit)) * 1e-6
        sigma_list.append(sigma)
    # sigma = np.average(sigma_list, weights=bins_list)
    sigma = np.mean(sigma_list)
    return sigma


def extract_a_b_depth_sigma(TOI: str, df: pd.DataFrame):
    """Extract a, b, depth, sigma to txt file.
    Args:
        TOI (str): TOI number
        df (pd.DataFrame): dataframe of the parameters of the planet
    """
    TOI = str(TOI)
    print("analysing: ", "TOI" + str(TOI))

    """惑星、主星の各パラメータを取得"""
    param_df = df[df["TOI"] == TOI]
    duration = param_df["Duration (hours)"].values[0] / 24
    period = param_df["Period (days)"].values[0]
    period_err = param_df["Period error"].values[0] / 24
    rp = (
        param_df["Planet Radius (R_Earth)"].values[0] * 0.00916794 * 696340000
    )  # translate to meter
    rp_err = (
        param_df["Planet Radius error"].values[0] * 0.00916794 * 696340000
    )  # translate to meter
    rs = (
        param_df["Stellar Radius (R_Sun)"].values[0] * 696340000
    )  # translate to meter
    rs_err = (
        param_df["Stellar Radius error"].values[0] * 696340000
    )  # translate to meter
    ms = (
        param_df["Stellar Mass (M_Sun)"].values[0] * 1.989e30
    )  # translate to kg
    ms_err = (
        param_df["Stellar Mass error"].values[0] * 1.989e30
    )  # translate to kg
    mp = (
        param_df["Predicted Mass (M_Earth)"].values[0] * 1.989e30 / 333030
    )  # translate to kg

    # periodが取得できない場合、処理を中断する
    if np.isnan(period):
        print(f"not found period:{TOI}")
        with open("no_period_found_20230409.txt", "a") as f:
            print(f"{TOI}", file=f)
        return

    lc = get_pdcsap_lc_from_mast(TOI)

    # calcuate a_m, b
    a_m, b = calc_b_and_a(period, period_err, ms, mp, rs, rp, duration)

    # get depth and
    pdb.set_trace()
    # calculate sigma
    sigma = calc_sigma(param_df, period, duration, lc)
    # save TOI, a_m, b, depth, sigma
    with open("./TOI_a_b_depth_sigma.txt", "a") as f:
        f.write(
            str(TOI)
            + " "
            + str(a_m)
            + " "
            + str(b)
            + " "
            + str(depth)
            + " "
            + str(sigma)
            + "\n"
        )


def modify_a_b_depth_sigma_txt(
    TOI: str, df: pd.DataFrame, txt_df: pd.DataFrame
) -> pd.DataFrame:
    """Modify a, b, depth, sigma to txt file.
    Args:
        TOI (str): TOI number
        df (pd.DataFrame): dataframe of the parameters of the planet
        txt_df (pd.DataFrame): dataframe of a, b, depth, sigma of the planet
    """

    TOI = str(TOI)
    print("modifing: ", "TOI" + str(TOI))

    """惑星、主星の各パラメータを取得"""
    param_df = df[df["TOI"] == TOI]
    duration = param_df["Duration (hours)"].values[0] / 24
    period = param_df["Period (days)"].values[0]
    period_err = param_df["Period error"].values[0] / 24
    rp = (
        param_df["Planet Radius (R_Earth)"].values[0] * 0.00916794 * 696340000
    )  # translate to meter
    rp_err = (
        param_df["Planet Radius error"].values[0] * 0.00916794 * 696340000
    )  # translate to meter
    rs = (
        param_df["Stellar Radius (R_Sun)"].values[0] * 696340000
    )  # translate to meter
    rs_err = (
        param_df["Stellar Radius error"].values[0] * 696340000
    )  # translate to meter
    ms = (
        param_df["Stellar Mass (M_Sun)"].values[0] * 1.989e30
    )  # translate to kg
    ms_err = (
        param_df["Stellar Mass error"].values[0] * 1.989e30
    )  # translate to kg
    mp = (
        param_df["Predicted Mass (M_Earth)"].values[0] * 1.989e30 / 333030
    )  # translate to kg

    # periodが取得できない場合、処理を中断する
    if np.isnan(period):
        print(f"not found period:{TOI}")
        with open("no_period_found_20230409.txt", "a") as f:
            print(f"{TOI}", file=f)
        return

    # calcuate a_m, b
    a_m, b = calc_b_and_a(period, period_err, ms, mp, rs, rp, duration)
    # modify a_m and b
    txt_df.loc[txt_df["TOI"] == float(TOI), "a_m"] = a_m
    txt_df.loc[txt_df["TOI"] == float(TOI), "b"] = b

    return txt_df


def extract_a_b_depth_sigma_wrapper(args):
    """Wrapper function of extract_a_b_depth_sigma
    Args:
        args (List): [TOI, df, txt_df]
    """

    return extract_a_b_depth_sigma(*args)


def get_pdcsap_lc_from_mast(TOI: str) -> lk.LightCurve:
    """Get PDCSAP light curve from MAST
    Args:
        TOI (str): TOI number
    """

    # lightkurveを用いてSPOCが作成した2min cadenceの全セクターのライトカーブをダウンロードする
    search_result = lk.search_lightcurve(
        f"TOI {TOI[:-3]}", mission="TESS", cadence="short", author="SPOC"
    )

    try:
        lc_collection = search_result.download_all()
    except AttributeError:
        lc_collection = search_result[0].download()

    if lc_collection is None:
        with open("no_data_found_20230409.txt", "a") as f:
            print(f"{TOI}", file=f)
        return

    # 全てのライトカーブを結合し、fluxがNaNのデータ点は除去する
    try:
        lc = lc_collection.stitch().remove_nans()
    except lk.utils.LightkurveError:
        search_result = lk.search_lightcurve(
            f"TOI {TOI[:-3]}", mission="TESS", cadence="short", author="SPOC"
        )
        lc_collection = search_result.download_all()
        lc = lc_collection.stitch().remove_nans()

    return lc


def get_lc_from_mast(TOI: str, AFTER_PLD_DATA_DIR, lc_type="sap"):
    """lightkurveを用いてSPOCが作成した2min cadenceの全セクターのライトカーブをダウンロードする"""
    search_result = lk.search_lightcurve(
        f"TOI {TOI[:-3]}", mission="TESS", cadence="short", author="SPOC"
    )

    try:
        lc_collection = search_result.download_all()
    except AttributeError:
        lc_collection = search_result[0].download()

    if lc_collection is None:
        with open("no_data_found_20230409.txt", "a") as f:
            print(f"{TOI}", file=f)
        return

    # lc_collectionがlistの場合、sectorごとに処理を行う
    for lc in lc_collection:
        # sectorが24以上の場合、処理を中断する
        if lc.sector >= 56:
            continue
        if os.path.exists(f"{AFTER_PLD_DATA_DIR}/TOI{TOI}_{lc.sector}.csv"):
            print(f"TOI{TOI}_{lc.sector}.csv already exists")
            continue
        # sap fluxに変換
        if lc_type == "sap":
            lc.flux = lc.sap_flux
            lc.flux_err = lc.sap_flux_err
            corrected_lc = correct_sap_lc(lc)
        else:
            corrected_lc = lc
        # corrected_lcをAFTER_PLD_DATA_DIRに保存
        os.makedirs(AFTER_PLD_DATA_DIR, exist_ok=True)
        corrected_lc.to_csv(
            f"{AFTER_PLD_DATA_DIR}/TOI{TOI}_{lc.sector}.csv", overwrite=True
        )


def correct_sap_lc(lc):
    # 補正のためのCROWDSAP、FLFRCSAPを定義
    CROWDSAP = lc.hdu[1].header["CROWDSAP"]
    FLFRCSAP = lc.hdu[1].header["FLFRCSAP"]
    median_flux = np.median(lc.flux.value)
    excess_flux = (1 - CROWDSAP) * median_flux
    flux_removed = lc.flux.value - excess_flux
    flux_corr = flux_removed / FLFRCSAP
    flux_err_corr = lc.flux_err.value / FLFRCSAP
    lc_corr = lk.LightCurve(
        time=lc.time.value, flux=flux_corr, flux_err=flux_err_corr
    )

    return lc_corr


def calc_data_survival_rate(lc: lk.LightCurve, duration: float) -> float:
    """Calculate data survival rate
    Args:
        lc (lk.LightCurve): light curve
        duration (float): duration of the transit
    """

    data_n = len(lc.flux)
    try:
        max_data_n = duration * 5 / (2 / 60 / 24)
        data_survival_rate = (data_n / max_data_n) * 100
    except IndexError:
        data_survival_rate = 0
        return data_survival_rate
    print(f"{data_survival_rate:2f}% data usable")
    # max_data_n = (lc.time_original[-1]-lc.time_original[0])*24*60/2

    return data_survival_rate


def q_to_u_limb(q_arr: List) -> np.ndarray:
    """Convert q1 and q2 to u1 and u2
    Args:
        q_arr (List): [q1, q2]
    """
    q1 = q_arr[0]
    q2 = q_arr[1]
    u1 = np.sqrt(q1) * 2 * q2
    u2 = np.sqrt(q1) * (1 - 2 * q2)
    return np.array([u1, u2])


def set_params_batman(
    params_lm: lmfit.Parameters, p_names: List, limb_type="quadratic"
) -> batman.TransitParams:
    """Set batman parameters
    Args:
        params_lm (lmfit.Parameters): lmfit parameters
        p_names (List): parameter names
        limb_type (str, optional): limb darkening type. Defaults to "quadratic".
    """
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


def noring_params_setting(period: float, rp_rs: float) -> lmfit.Parameters:
    """Set parameters for no ring transit model.
    Args:
        period (float): orbital period
        rp_rs (float): planet radius / stellar radius
    """
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
        -0.5,
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
        0.5,
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


def ring_params_setting(
    no_ring_res: lmfit.minimizer.MinimizerResult,
    b: float,
    period: float,
    rp_rs: float,
    theta: float,
    phi: float,
    simulation: bool = False,
) -> lmfit.Parameters:
    """Set parameters for ring transit model.
    Args:
        no_ring_res (lmfit.minimizer.MinimizerResult): result of no ring transit model
        b (float): impact parameter
        period (float): orbital period
        rp_rs (float): planet radius / stellar radius
        theta (float): longitude of the ring
        phi (float): latitude of the ring
    Returns:
        lmfit.Parameters: parameters for ring transit model
    """

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
    params = set_params_lm(names, saturnlike_values, mins, maxes, vary_flags)

    return params


def set_params_lm(
    names: List, values: List, mins: List, maxes: List, vary_flags: List
) -> lmfit.Parameters:
    """Set parameters for lmfit.
    Args:
        names (List): parameter names
        values (List): parameter values
        mins (List): minimum values
        maxes (List): maximum values
        vary_flags (List): vary flags
    Returns:
        lmfit.Parameters: parameters for lmfit
    """

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


def set_transit_times(
    transit_time: float, period: float, lc: lk.LightCurve
) -> List:
    """
    make transit time list from mid transit time and orbital period.
    Args:
        transit_time (float): mid transit time.
        period (float): orbital period.
    Returns:
        List: the list of mid transit time.
    """

    transit_time_list = np.append(
        np.arange(transit_time, lc.time[-1].value, period),
        np.arange(transit_time, lc.time[0].value, -period),
    )
    transit_time_list = np.unique(transit_time_list)
    transit_time_list.sort()

    return transit_time_list


def ring_model(
    t: List, pdic: Dict, mcmc_params=None, mcmc_pvalues=None
) -> List:
    """calculate flux of ring transit model.
    Args:
        t (List): time.
        pdic (Dict): parameter dictionary.
        mcmc_pvalues (None, optional): mcmc parameter values. Defaults to None.
    Returns:
        List: flux of ring transit model.

    """
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


def ring_transitfit(
    params: lmfit.Parameters, t: List, flux: List, flux_err: List
) -> List:
    """fitting ring transit model.
    Args:
        params (lmfit.Parameters): parameters for lmfit.
        t (List): time.
        flux (List): flux.
        flux_err (List): flux error.
    Returns:
        List: residual.
    """

    model = ring_model(t, params.valuesdict())
    chi_square = np.sum(((flux - model) / flux_err) ** 2)
    print(chi_square)

    return (flux - model) / flux_err


def no_ring_transitfit(
    params: lmfit.Parameters,
    t: List,
    flux: List,
    flux_err: List,
    return_model=False,
):
    """fitting no ring transit model.
    Args:
        params (lmfit.Parameters): parameters for lmfit.
        t (List): time.
        flux (List): flux.
        flux_err (List): flux error.
        p_names (List): parameter names.
        return_model (bool, optional): return model or not. Defaults to False.
    Returns:
        List: residual.
    """
    params_batman = set_params_batman(params, list(params.valuesdict().keys()))
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


def no_ring_transit_and_polynomialfit(
    params: lmfit.Parameters,
    t: List,
    flux: List,
    flux_err: List,
    return_model=False,
) -> List:
    """fitting no ring transit model and polynomial.
    Args:
        params (lmfit.Parameters): parameters for lmfit.
        lc (lk.LightCurve): lightkurve object.
        return_model (bool, optional): return model or not. Defaults to False.
    Returns:
        List: residual.
    """
    params_batman = set_params_batman(params, list(params.valuesdict().keys()))
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
    model = transit_model * polynomialmodel
    # chi_square = np.sum(((flux - model) / flux_err) ** 2)
    # print(params)
    # print(chi_square)
    # visualize_plot_process(model, x, data, eps_data)
    if return_model:
        return model, transit_model, polynomialmodel
    else:
        return (flux - model) / flux_err


def transit_params_setting(rp_rs, period):
    """トランジットフィッティングパラメータの設定"""
    p_names = ["t0", "per", "rp", "a", "b", "ecc", "w", "q1", "q2"]
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
    mins = [-0.5, period * 0.8, 0.01, 1, 0, 0, 90, 0.0, 0.0]
    vary_flags = [True, False, True, True, True, False, False, True, True]
    return set_params_lm(p_names, values, mins, maxes, vary_flags)


def make_oc_diagram(
    t0list: List,
    t0errlist: List,
    transit_time_list: List,
    transit_time_error: float,
    TOInumber: str,
    SAVE_OC_DIAGRAM: str,
):
    """calculate observed transit time.
    Args:
        t0list (List): transit time list.
        t0errlist (List): transit time error list.
        num_list (List): number list.
        transit_time_list (List): transit time list.
        transit_time_error (float): transit time error.
    Returns:
        List: observed transit time.
    """
    # orbital periodとtransit timeから算出したtransit timeと、実際にフィッティングしたときのtransit timeの差を計算
    diff = t0list - transit_time_list
    transit_time_list = transit_time_list[~(diff == 0)]
    t0errlist = t0errlist[~(t0errlist == 0)]  # todo: この処理は正しいのか？

    # transit time variationを確認するためにO-C図を作成
    x = np.array(t0list)[~(diff == 0)]
    y = np.array(x - transit_time_list) * 24  # [days] > [hours]
    yerr = (
        np.sqrt(np.square(t0errlist) + np.square(transit_time_error)) * 24
    )  # [days] > [hours]

    # O-C図を保存
    plt.errorbar(x=x, y=y, yerr=yerr, fmt=".k")
    plt.xlabel("mid transit time[BJD] - 2457000")
    plt.ylabel("O-C(hrs)")
    plt.tight_layout()
    plt.savefig(f"{SAVE_OC_DIAGRAM}/{TOInumber}.png")
    plt.close()


def estimate_period(
    t0list: List,
    t0errlist: List,
    num_list: List,
    transit_time_list: List,
    TOInumber: str,
    SAVE_ESTIMATED_PER: str,
    period: float,
):
    diff = t0list - transit_time_list
    x = np.array(num_list)
    y = np.array(t0list)[~(diff == 0)]
    yerr = np.array(t0errlist)[~(diff == 0)]

    # 線形回帰
    try:
        res = linregress(x, y)
    except ValueError:
        print("ValueError: Inputs must not be empty.")
        pdb.set_trace()

    # 傾きを推定したperiodとする
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
        plt.savefig(f"{SAVE_ESTIMATED_PER}/{TOInumber}.png")
        # plt.show()
        plt.close()
        return estimated_period, ts * res.stderr
    else:
        print("np.isnan(ts*res.stderr) == True")
        estimated_period = period
        return estimated_period, None


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
                for p_name in list(params.valuesdict().keys()):
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
                args=(t, flux, flux_err),
                max_nfev=10000,
                nan_policy="omit",
                method="least_squares",
            )
            if res.params["t0"].stderr != None:
                if np.isfinite(res.params["t0"].stderr):
                    # and res.redchi < 10:
                    # if res.redchi < 10:
                    red_redchi = res.redchi - 1
                    best_res_dict[red_redchi] = res
        if len(best_res_dict) == 0:
            print(lmfit.fit_report(res))
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
    each_lc.flux = each_lc.flux / poly_model
    each_lc.flux_err = each_lc.flux_err / poly_model

    return each_lc


def stack_lc_from_csv(loaddir):
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


def plot_ring(
    TOInumber, ring_res, rp_rs, rin_rp, rout_rin, b, theta, phi, file_name
):
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

    # calculte of ellipse of rings
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
    ax.set_title(f"chisq={str(ring_res.redchi)[:6]}")
    plt.axis("scaled")
    ax.set_aspect("equal")
    plt.savefig(file_name, bbox_inches="tight")
    plt.close()
    # plt.show()


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

    # 各リストの長さ
    length = len(binned_time[1:])

    # nanを含むインデックスを特定
    nan_indices = np.where(
        np.isnan(binned_time[1:])
        | np.isnan(binned_flux)
        | np.isnan(binned_flux_err)
    )[0]

    # nanを含むインデックス以外の要素を抽出
    binned_time_clean = [
        binned_time[i + 1] for i in range(length) if i not in nan_indices
    ]
    binned_flux_clean = [
        binned_flux[i] for i in range(length) if i not in nan_indices
    ]
    binned_flux_err_clean = [
        binned_flux_err[i] for i in range(length) if i not in nan_indices
    ]

    return (
        np.array(binned_time_clean),
        np.array(binned_flux_clean),
        np.array(binned_flux_err_clean),
    )


def make_simulation_data(
    period, b, rp_rs, bin_error, theta=45, phi=0, r_in=1.01, r_out=1.70
):
    """make_simulation_data"""
    t = np.linspace(-0.1, 0.1, 500)
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

    # tをdurationに合わせて再計算
    res = transit_fitting(each_lc, rp_rs, period)
    # durationを抜き出す
    a_rs = res.params["a"].value
    b = res.params["b"].value
    inc = np.arccos(b / a_rs)
    rp_rs = res.params["rp"].value
    duration = (period / np.pi) * np.arcsin(
        (1 / a_rs)
        * (np.sqrt(np.square(1 + rp_rs) - np.square(b)) / np.sin(inc))
    )
    # プラスマイナスduration×0.8の時間を取り出す
    t = np.linspace(duration * (-0.6), duration * 0.6, 500)
    ymodel = ring_model(t, pdic_saturnlike)
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


def get_params_from_table(df: pd.DataFrame, TOI: int):
    # Filter the DataFrame to get the rows matching the TOI value
    matched_rows = df[df["TOI"] == TOI]

    # Check if any rows matched the TOI value
    if matched_rows.empty:
        raise ValueError("No rows found for the given TOI value")

    # Extract the parameters from the matched rows

    duration = matched_rows["Duration (hours)"].values[0] / 24
    period = matched_rows["Period (days)"].values[0]
    transit_time = matched_rows["Transit Epoch (BJD)"].values[0] - 2457000.0
    transit_time_error = matched_rows["Transit Epoch error"].values[0]
    rp = matched_rows["Planet Radius (R_Earth)"].values[0] * 0.00916794
    rs = matched_rows["Stellar Radius (R_Sun)"].values[0]
    # ms = matched_rows["Stellar Mass (M_Sun)"].values[0] * 1.989e30
    # mp = matched_rows["Predicted Mass (M_Earth)"].values[0] * 1.989e30 / 333030

    rp_rs = rp / rs

    # Return the extracted parameters as a dictionary
    return {
        "duration": duration,
        "period": period,
        "transit_time": transit_time,
        "transit_time_error": transit_time_error,
        "rp": rp,
        "rs": rs,
        # "ms": ms,
        # "mp": mp,
        "rp_rs": rp_rs,
    }


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


def get_rp_rs(
    depth: float, b: float, period: float, theta: float, phi: float
) -> float:
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


def process_bin_error(bin_error, b, rp_rs, theta, phi, period, min_flux):
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
            args=(t, flux, flux_err),
            max_nfev=1000,
            method="least_squares",
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
                args=(t, flux, flux_err),
                max_nfev=100,
                method="least_squares",
            )
        except ValueError:
            print("Value Error")
            print(list(params.values()))
            continue
        red_redchi = ring_res.redchi - 1
        best_ring_res_dict[red_redchi] = ring_res
    ring_res = sorted(best_ring_res_dict.items())[0][1]
    fig = plt.figure()
    ax_lc = fig.add_subplot(2, 1, 1)  # for plotting transit model and data
    ax_re = fig.add_subplot(2, 1, 2)  # for plotting residuals
    ring_flux_model = ring_transitfit(
        ring_res.params,
        t,
        flux,
        flux_err,
        return_model=True,
    )
    noring_flux_model = no_ring_transitfit(
        no_ring_res.params,
        t,
        flux,
        flux_err,
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
    ax_lc.plot(t, ring_flux_model, label="Model w/ ring", color="blue")
    ax_lc.plot(t, noring_flux_model, label="Model w/o ring", color="red")
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
        f"./target_selection/figure/b_{b}/{theta}deg_{phi}deg",
        exist_ok=True,
    )
    plt.savefig(
        f"./target_selection/figure/b_{b}/{theta}deg_{phi}deg/{min_flux}_{bin_error}.png"
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
        f"./target_selection/data/b_{b}/{theta}deg_{phi}deg",
        exist_ok=True,
    )
    with open(
        f"./target_selection/data/b_{b}/{theta}deg_{phi}deg/{min_flux}_{bin_error}.txt",
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


def calc_ring_res(
    m,
    no_ring_res,
    t,
    flux_data,
    flux_err_data,
    period,
    TOInumber,
    noringnames,
    ring_res_p_dir,
    plt_transit_dir,
    plt_ringfig_dir,
):
    np.random.seed(int(np.random.rand() * 1000))

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

    q1 = np.random.uniform(0.1, 0.9)
    q2 = np.random.uniform(0.1, 0.9)
    t0 = 0
    rp_rs = np.random.uniform(
        no_ring_res.params.valuesdict()["rp"] * 0.9,
        no_ring_res.params.valuesdict()["rp"] * 1.1,
    )
    a_rs = np.random.uniform(
        no_ring_res.params.valuesdict()["a"],
        no_ring_res.params.valuesdict()["a"] * 1.5,
    )
    b = np.random.uniform(0.0, 1.0)
    norm = 1
    theta = np.random.uniform(-np.pi / 2, np.pi / 2)
    phi = np.random.uniform(-np.pi / 2, np.pi / 2)
    tau = 1
    r_in = 1.01
    r_out = np.random.uniform(1.02, 1.70)

    values = [
        q1,
        q2,
        t0,
        period,
        rp_rs,
        a_rs,
        b,
        norm,
        theta,
        phi,
        tau,
        r_in,
        r_out,
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
        -np.pi / 2,
        -np.pi / 2,
        1.00,
        1.00,
        1.00,
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
        1.2,
        1.1,
        np.pi / 2,
        np.pi / 2,
        1.0,
        2.45,
        1.80,
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
        False,
        True,
        False,
        False,
        False,
        False,
    ]

    params = set_params_lm(names, values, mins, maxes, vary_flags)
    params_df = pd.DataFrame(
        list(zip(values, mins, maxes)),
        columns=["values", "mins", "maxes"],
        index=names,
    )
    vary_dic = dict(zip(names, vary_flags))
    params_df = params_df.join(
        pd.DataFrame.from_dict(
            vary_dic, orient="index", columns=["vary_flags"]
        )
    )
    pdic = params_df["values"].to_dict()

    max_attempts = 10
    attempt = 0

    while attempt < max_attempts:
        try:
            ring_res = lmfit.minimize(
                ring_transitfit,
                params,
                args=(t, flux_data, flux_err_data),
                max_nfev=1000,
                method="least_squares",
                nan_policy="omit",
            )
            break  # 最適化が成功した場合はループを終了
        except ValueError:
            print("value error")
        attempt += 1

    if attempt >= max_attempts:
        print("最大試行回数を超えました。最適化に失敗しました。")
    if ring_res.params["r_out"].stderr != None and np.isfinite(
        ring_res.params["r_out"].stderr
    ):
        F_obs = (
            (no_ring_res.chisqr - ring_res.chisqr)
            / (ring_res.nvarys - no_ring_res.nvarys)
        ) / (ring_res.chisqr / (ring_res.ndata - ring_res.nvarys - 1))
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
        # best_ring_res_dict[F_obs] = ring_res

        # 　csvにfitting parametersを書き出し
        input_df = pd.DataFrame.from_dict(
            params.valuesdict(), orient="index", columns=["input_value"]
        )
        output_df = pd.DataFrame.from_dict(
            ring_res.params.valuesdict(),
            orient="index",
            columns=["output_value"],
        )
        input_df = input_df.applymap(lambda x: "{:.6f}".format(x))
        output_df = output_df.applymap(lambda x: "{:.6f}".format(x))
        result_df = input_df.join(
            (output_df, pd.Series(vary_flags, index=names, name="vary_flags"))
        )
        result_df.to_csv(
            f"{ring_res_p_dir}/{TOInumber}_{F_obs:.0f}_{m}.csv",
            header=True,
            index=False,
        )
        plot_ring(
            TOInumber,
            ring_res,
            rp_rs=ring_res.params["rp_rs"].value,
            rin_rp=ring_res.params["r_in"].value,
            rout_rin=ring_res.params["r_out"].value,
            b=ring_res.params["b"].value,
            theta=ring_res.params["theta"].value,
            phi=ring_res.params["phi"].value,
            file_name=f"{plt_ringfig_dir}/{TOInumber}_{F_obs:.0f}_{m}.pdf",
        )

        fig = plt.figure()
        ax_lc = fig.add_subplot(2, 1, 1)  # for plotting transit model and data
        ax_re = fig.add_subplot(2, 1, 2)  # for plotting residuals
        ring_flux_model = ring_transitfit(
            ring_res.params,
            t,
            flux_data,
            flux_err_data,
        )
        noring_flux_model = no_ring_transitfit(
            no_ring_res.params,
            t,
            flux_data,
            flux_err_data,
            return_model=True,
        )
        ax_lc.errorbar(
            x=t,
            y=flux_data,
            yerr=flux_err_data,
            color="black",
            marker=".",
            linestyle="None",
            zorder=1,
        )
        ax_lc.plot(
            t, ring_flux_model, label="Model w/ ring", color="blue", zorder=3
        )
        ax_lc.plot(
            t, noring_flux_model, label="Model w/o ring", color="red", zorder=2
        )
        residuals_ring = flux_data - ring_flux_model
        residuals_no_ring = flux_data - noring_flux_model
        ax_re.plot(
            t, residuals_ring, color="blue", alpha=0.3, marker=".", zorder=2
        )
        ax_re.plot(
            t, residuals_no_ring, color="red", alpha=0.3, marker=".", zorder=1
        )
        ax_re.plot(t, np.zeros(len(t)), color="black", zorder=3)
        ax_lc.legend()
        # ax_lc.set_title(f'w/ chisq:{ring_res.chisqr:.0f}/{ring_res.nfree:.0f} w/o chisq:{no_ring_res.chisqr:.0f}/{no_ring_res.nfree:.0f}')
        ax_lc.set_title(f"{TOInumber}, p={p_value:.2e}")
        plt.tight_layout()
        # os.makedirs(f'./simulation/transit_fit/{TOInumber}', exist_ok=True)
        plt.savefig(f"{plt_transit_dir}/{TOInumber}_{F_obs:.0f}_{m}.pdf")
        plt.close()

        return {"res": ring_res, "F_obs": F_obs}
    else:
        return {"res": ring_res, "F_obs": -999}


def calc_bin_std(lc, TOInumber):
    std_list = []
    std_err_list = []
    phase_list = []
    min_list = lc.bin(bins=500).remove_nans().time.value
    for i in range(len(min_list)):
        try:
            part_lc = lc[lc.time.value > min_list[i]]
            part_lc = part_lc[part_lc.time.value < min_list[i + 1]]
            if len(part_lc) == 0:
                continue
            std_list.append(part_lc.flux.std())
            std_err_list.append(part_lc.flux_err.std())
            phase_list.append(part_lc.time[0].value)
        except IndexError:
            pass
    # print(len(std_list))
    # print(len(phase_list))
    fig = plt.figure()
    ax2 = fig.add_subplot(2, 1, 1)  # for plotting transit model and data
    ax1 = fig.add_subplot(2, 1, 2)  # for plotting residuals
    lc.errorbar(ax=ax1, color="black", marker=".", zorder=1, label="data")
    ax2.errorbar(x=phase_list, y=std_list, yerr=std_err_list, fmt=".k")
    # ax2.set_xlabel('orbital phase')
    ax2.set_ylabel("Flux std")
    ax2.set_title(TOInumber)
    plt.savefig(
        f"/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/fitting_result/figure/bin_std/{TOInumber}.png"
    )
    # plt.show()
    d = {"phase": phase_list, "bin_std": std_list}
    pd.DataFrame(data=d).to_csv(
        f"/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/fitting_result/data/bin_std/{TOInumber}.csv"
    )
    plt.close()


def calc_ring_res_wrapper(args):
    """引数をアンパックして渡す関数です。"""
    return calc_ring_res(*args)


def main():
    args = sys.argv
    # csvfile = './folded_lc_data/TOI2403.01.csv'
    # done_TOIlist = os.listdir('./lmfit_res/sap_0605_p0.01ult/transit_fit') #ダブリ解析防止
    # oridf = pd.read_csv('./exofop_tess_tois.csv')
    oridf = pd.read_csv("./exofop_tess_tois_20230526.csv")
    # df = oridf[oridf["Planet SNR"] > 100]
    df = oridf
    """
    df['Rjup'] = df['Planet Radius (R_Earth)']/11.209
    plt.scatter(x=df['Period (days)'], y = df['Rjup'], color='k')
    plt.xlabel('Orbital Period(day)')
    plt.ylabel(r'Planet Radius ($R_{J}$)')
    plt.gca().yaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
    plt.gca().xaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
    plt.xscale('log')
    plt.minorticks_on()
    plt.show()
    """
    df["TOI"] = df["TOI"].astype(str)
    df = df.sort_values("Planet SNR", ascending=False)

    HOMEDIR = os.getcwd()
    folded_lc_dir = f"{HOMEDIR}/folded_lc_data/0605_p0.05"
    done_fit_report_dir = f"{HOMEDIR}/lmfit_res/sap_0605_p0.05/fit_report"

    toidf = pd.read_csv(f"{HOMEDIR}/below_p_0.05_TOIs.csv")
    TOIlist = toidf["TOI"].values
    TOIlist = [str(s) for s in TOIlist]
    # TOIlist = [s.strip(".csv").strip("TOI") for s in os.listdir(folded_lc_dir)]

    try:
        done_list = [
            s.lstrip("TOI").strip(".txt")
            for s in os.listdir(done_fit_report_dir)
            if "TOI" in s
        ]
        TOIlist = [item for item in TOIlist if item not in done_list]
    except FileNotFoundError:
        pass
    print(len(TOIlist))
    for TOI in TOIlist[int(args[1]) : int(args[1]) + 1]:
        print(TOI)
        pdb.set_trace()
        TOInumber = "TOI" + TOI
        param_df = df[df["TOI"] == TOI]

        csvfile = f"{HOMEDIR}/folded_lc_data/0605_p0.05/{TOInumber}.csv"
        load_dur_per_file = f"{HOMEDIR}/folded_lc_data/fit_report_0605_p0.05/{TOInumber}_folded.txt"
        noring_res_p_dir = f"{HOMEDIR}/lmfit_res/sap_0605_p0.01/fit_p_data/no_ring_model/{TOInumber}"
        ring_res_p_dir = f"{HOMEDIR}/lmfit_res/sap_0605_p0.01/fit_p_data/ring_model/{TOInumber}"
        save_fit_report_dir = f"{HOMEDIR}/lmfit_res/sap_0605_p0.01/fit_report/"
        plt_ringfig_dir = (
            f"{HOMEDIR}/lmfit_res/sap_0605_p0.01/illustration/{TOInumber}"
        )
        plt_transit_dir = (
            f"{HOMEDIR}/lmfit_res/sap_0605_p0.01/transit_fit/{TOInumber}"
        )

        for dir in [
            noring_res_p_dir,
            ring_res_p_dir,
            save_fit_report_dir,
            plt_ringfig_dir,
            plt_transit_dir,
        ]:
            os.makedirs(dir, exist_ok=True)
        # parameters setting
        try:
            with open(
                load_dur_per_file,
                "r",
            ) as f:
                # durationline = f.readlines()[-5:].split(' ')[-1]
                durationline, _, _, periodline, _ = f.readlines()[-5:]
                durationline = durationline.split(" ")[-1]
                duration = np.float(durationline)
                periodline = periodline.split(" ")[-1]
                period = np.float(periodline)
        except FileNotFoundError:
            with open("./folded.txt_FileNotFoundError.txt", "a") as f:
                f.write(TOInumber + "\n")
            try:
                duration = param_df["Duration (hours)"].values[0] / 24
                period = param_df["Period (days)"].values[0]
            except IndexError:
                with open("./noparameterindf.txt", "a") as f:
                    f.write(TOInumber + "\n")
                continue
        try:
            rp = (
                param_df["Planet Radius (R_Earth)"].values[0] * 0.00916794
            )  # translate to Rsun
            rs = param_df["Stellar Radius (R_Sun)"].values[0]
        except IndexError:
            with open("./noparameterindf.txt", "a") as f:
                f.write(TOInumber + "\n")
            continue
        rp_rs = rp / rs

        # csvfile = f'/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/fitting_result/data/folded_lc/4poly/obs_t0/csv/{TOInumber}.csv'
        try:
            folded_table = ascii.read(csvfile)
        except FileNotFoundError:
            with open("./csvfile_FileNotFoundError.txt", "a") as f:
                f.write(TOInumber + ",")
            continue
        folded_lc = lk.LightCurve(data=folded_table)

        folded_lc = folded_lc[
            (folded_lc.time.value < duration * 0.7)
            & (folded_lc.time.value > -duration * 0.7)
        ]
        # calc_bin_std(folded_lc, TOInumber)
        if len(folded_lc.time) < 500:
            t = folded_lc.time.value
            flux_data = folded_lc.flux.value
            flux_err_data = folded_lc.flux_err.value
        else:
            t, flux_data, flux_err_data = binning_lc(folded_lc)
        if np.sum(np.isnan([t, flux_data, flux_err_data])) != 0:
            with open("./NaN_values_detected.txt", "a") as f:
                print(t, flux_data, flux_err_data)
                f.write(TOInumber + ",")
                continue
        # 　no ring model fitting by minimizing chi_square
        best_res_dict = {}
        for n in range(50):
            noringnames = ["t0", "per", "rp", "a", "b", "ecc", "w", "q1", "q2"]
            if np.isnan(rp_rs):
                noringvalues = [
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
            else:
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
                args=(t, flux_data, flux_err_data),
                max_nfev=1000,
                method="least_squares",
                nan_policy="omit",
            )
            if no_ring_res.params["t0"].stderr != None:
                if (
                    np.isfinite(no_ring_res.params["t0"].stderr)
                    and no_ring_res.redchi < 10
                ):
                    red_redchi = no_ring_res.redchi - 1
                    best_res_dict[red_redchi] = no_ring_res
        no_ring_res = sorted(best_res_dict.items())[0][1]
        input_df = pd.DataFrame.from_dict(
            no_ring_params.valuesdict(),
            orient="index",
            columns=["input_value"],
        )
        output_df = pd.DataFrame.from_dict(
            no_ring_res.params.valuesdict(),
            orient="index",
            columns=["output_value"],
        )
        input_df = input_df.applymap(lambda x: "{:.6f}".format(x))
        output_df = output_df.applymap(lambda x: "{:.6f}".format(x))
        result_df = input_df.join(
            (
                output_df,
                pd.Series(
                    noringvary_flags, index=noringnames, name="vary_flags"
                ),
            )
        )
        result_df.to_csv(
            f"{noring_res_p_dir}/{TOInumber}_{no_ring_res.redchi:.2f}.csv",
            header=True,
            index=False,
        )

        ###ring model fitting by minimizing chi_square###
        res_list = []
        for m in range(250):
            res = calc_ring_res(
                m,
                no_ring_res,
                t,
                flux_data,
                flux_err_data,
                period,
                TOInumber,
                noringnames,
                ring_res_p_dir,
                plt_transit_dir,
                plt_ringfig_dir,
            )
            res_list.append(res)
        pdb.set_trace()
        """
        m = range(250)
        
        src_datas = list(
            map(
                lambda x: [
                    x,
                    no_ring_res,
                    t,
                    flux_data,
                    flux_err_data,
                    period,
                    TOInumber,
                    noringnames,
                    ring_res_p_dir,
                    plt_transit_dir,
                    plt_ringfig_dir,
                ],
                m,
            )
        )
        batch_size = 3  # バッチサイズの定義
        batches = [
            src_datas[i : i + batch_size]
            for i in range(0, len(src_datas), batch_size)
        ]
        with futures.ProcessPoolExecutor(max_workers=3) as executor:
            future_list = []
            for batch in batches:
                future_list.extend(
                    executor.map(calc_ring_res_wrapper, batch, timeout=None)
                )
        """
        # future_res_list = [f.result() for f in future_list]
        print(future_list)
        for i, fit_res in enumerate(future_list):
            if i > 0:
                if fit_res["F_obs"] > F_obs:
                    ring_res = fit_res["res"]
                    F_obs = fit_res["F_obs"]
                    print(ring_res)
                    print(F_obs)
                else:
                    pass
            else:
                ring_res = fit_res["res"]
                F_obs = fit_res["F_obs"]
                print(ring_res)
                print(F_obs)
        # F_obs = ( (no_ring_res.chisqr-ring_res.chisqr)/ (ring_res.nvarys-no_ring_res.nvarys) ) / ( ring_res.chisqr/(ring_res.ndata-ring_res.nvarys-1) )
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
        with open(
            f"{save_fit_report_dir}/{TOInumber}.txt",
            "a",
        ) as f:
            print("no ring transit fit report:\n", file=f)
            print(lmfit.fit_report(no_ring_res), file=f)
            print("ring transit fit report:\n", file=f)
            print(lmfit.fit_report(ring_res), file=f)
            print(f"\nF_obs: {F_obs}", file=f)
            print(f"\np_value: {p_value}", file=f)


def main2():
    oridf = pd.read_csv("./exofop_tess_tois.csv")
    df = oridf[oridf["Planet SNR"] > 100]
    df["TOI"] = df["TOI"].astype(str)
    df = df.sort_values("Planet SNR", ascending=False)
    df["TOI"] = df["TOI"].astype(str)

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
    param_df = df[df["TOI"] == TOI]
    period = param_df["Period (days)"].values[0]

    b_list = [
        0,
        0.2,
        0.4,
        0.6,
        0.8,
        1,
        0.1,
        0.3,
        0.5,
        0.7,
        0.9,
    ]
    min_flux_list = [
        0.99,
        0.98,
        0.97,
        0.96,
        0.95,
        0.995,
        0.994,
        0.993,
        0.992,
        0.991,
        0.94,
        0.93,
        0.92,
        0.91,
        0.90,
        0.999,
        0.998,
        0.997,
        0.996,
    ]
    theta_list = [
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
        3,
        55,
        60,
        65,
        70,
        75,
        80,
        85,
        -5,
        -10,
        -15,
        -20,
        -25,
        -30,
        -35,
        -40,
        -45,
        -50,
        -55,
        -60,
        -65,
        -70,
        -75,
        -80,
        -85,
    ]
    phi_list = [
        0,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
        55,
        60,
        65,
        70,
        75,
        80,
        85,
        -5,
        -10,
        -15,
        -20,
        -25,
        -30,
        -35,
        -40,
        -45,
        -50,
        -55,
        -60,
        -65,
        -70,
        -75,
        -80,
        -85,
    ]
    b = b_list[9]
    for theta in theta_list:
        for phi in phi_list:
            for min_flux in min_flux_list:
                bin_error_list = np.arange(0.0001, 0.0041, 0.0001)
                bin_error_list = np.around(bin_error_list, decimals=4)

                new_bin_error_list = []

                for bin_error in bin_error_list:
                    if os.path.isfile(
                        f"./target_selection/figure/b_{b}/{theta}deg_{phi}deg/{min_flux}_{bin_error}.png"
                    ):
                        print(
                            f"b_{b}/{theta}deg_{phi}deg/{min_flux}_{bin_error}.png is exist."
                        )
                        continue
                    else:
                        new_bin_error_list.append(bin_error)

                if len(new_bin_error_list) == 0:
                    continue

                # get rp_rs value from transit depth
                rp_rs = get_rp_rs(min_flux, b, period, theta, phi)
                """
                for bin_error in new_bin_error_list:
                    process_bin_error(
                        bin_error, b, rp_rs, theta, phi, period, min_flux
                    )
                """
                src_datas = list(
                    map(
                        lambda x: [
                            x,
                            b,
                            rp_rs,
                            theta,
                            phi,
                            period,
                            min_flux,
                        ],
                        new_bin_error_list,
                    )
                )
                print(b, theta, phi, min_flux, bin_error)
                batch_size = 3  # バッチサイズの定義
                batches = [
                    src_datas[i : i + batch_size]
                    for i in range(0, len(src_datas), batch_size)
                ]
                with futures.ProcessPoolExecutor(max_workers=3) as executor:
                    for batch in batches:
                        future_list = executor.map(
                            process_bin_error_wrapper, batch, timeout=None
                        )


if __name__ == "__main__":
    main()
