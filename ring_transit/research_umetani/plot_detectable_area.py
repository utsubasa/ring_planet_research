import glob
import os
import pdb
import pickle
import re
import sys
from concurrent import futures
from multiprocessing import Pool, cpu_count

import lightkurve as lk
import matplotlib.patches as patches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from astropy.io import ascii
from scipy import integrate
from scipy.interpolate import interp1d
from tqdm import tqdm


def plot_theta_phi(b, depth, sigma):
    print(f"{depth} {sigma} {b}")
    # theta, phiの各フォルダから同名のファイルを読み込む
    filelist = glob.glob(
        f"{HOMEDIR}/depth_error/data/{b}/*/{depth}_{sigma}.txt"
    )
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 軸範囲を設定
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    for file in filelist:
        thetaphi = file.split("/")[-2]
        theta = float(thetaphi.split("_")[0].split("deg")[0])
        phi = float(thetaphi.split("_")[-1].split("deg")[0])
        with open(file, "r") as f:
            p_value_line = f.readlines()[-1]
            p_value = p_value_line.split(": ")[-1]
            p_value = float(p_value)
        # plot area
        if p_value <= 0.01:
            ax.scatter(theta, phi, fc="r", s=150, zorder=2)
        elif p_value <= 0.05:
            ax.scatter(theta, phi, fc="orange", s=150, zorder=1)
    ax.set_title(f"b, depth, sigma: {b[2:]}, {depth}, {sigma}")
    plt.xlabel("theta")
    plt.ylabel("phi")
    os.makedirs(
        f"{HOMEDIR}/depth_error/figure/{b}/theta_phi_axis/", exist_ok=True
    )
    plt.savefig(
        f"{HOMEDIR}/depth_error/figure/{b}/theta_phi_axis/pvalue_area_{depth}_{sigma}.png"
    )
    plt.close()


def plot_theta_phi_wrapper(args):
    plot_theta_phi(*args)


def is_point_below_line(x, y, interp_func):
    # 点が線の下にあるかどうかを判定
    if y < interp_func(x):
        return True
    else:
        return False


def calc_sigma_eq(file, depth):
    # P valueを求めている式の逆関数を解き、必要なF_obsの値を取得
    """
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
    """
    F_obs = 65
    # F_obsの式の逆関数をとき、必要なchi-squareの値を取得
    """
    F_obs = (
        (no_ring_res.chisqr + ring_model_chisq - ring_model_chisq)
        / (ring_res.nvarys - no_ring_res.nvarys)
    ) / (ring_model_chisq / (ring_res.ndata - ring_res.nvarys - 1))
        = (no_ring_res.chisqr / (9 - 6) ) / (500 / (500 - 9 - 1))
        = (no_ring_res.chisqr / 3 ) / (500 / 490 )
    
    no_ring_res.chisqr = 3 * F_obs * 1.02040816
    """
    scaled_chisq = F_obs * 3 * 1.02040816
    # chi-square=の式がdepth,sigmaの2次元でのプロットすべき式になる
    ref_chisq = get_chisq(file)
    A = scaled_chisq / ref_chisq
    sigma = REF_SIGMA * ((((1 - depth) / (1 - REF_DEPTH)) ** 2.5) / A) ** 0.5

    return sigma


def plot_target_area(r_out_list, b, theta, phi, df):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for r_out in r_out_list:
        file = f"{HOMEDIR}/target_selection/data/{b}/{r_out}/{theta}deg_{phi}deg/{REF_DEPTH}_{REF_SIGMA}.txt"
        # calc_sigma_eqを使ってplt.plotするためのsigmaを求める
        depth = np.linspace(0.95, 0.9999, 200)
        sigma = calc_sigma_eq(file, depth)
        # sigma, depthをdepthをキーにしてソートする
        sigma_depth_dict = dict(zip(sigma, depth))
        sigma_depth_dict = sorted(sigma_depth_dict.items(), key=lambda x: x[1])
        sigma, depth = zip(*sigma_depth_dict)
        plt.plot(
            depth,
            sigma,
            color="red",
            zorder=3,
            label=r"$r_{\rm{out}}$" + f"={r_out}, p=0.0026",
        )

    # /Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/oversn100_tois_depth_sigma.csvからdfを作成してプロット
    """
    over100_df = pd.read_csv(f"{HOMEDIR}/oversn100_tois_depth_sigma.csv")
    ax.scatter(
        over100_df.depth,
        over100_df.sigma,
        color="red",
        zorder=2,
        s=15,
    )
    """

    # dfからdepthが0.999以下で0.95以上の行を取得
    df = df[(df["depth"] <= 0.999) & (df["depth"] >= 0.95)]

    # df["sigma"] == 0の行を削除
    df = df[df["sigma"] != 0]

    ax.scatter(
        df.depth,
        df.sigma,
        color="black",
        zorder=1,
        s=15,
    )
    plt.title(f"b: {b}, theta: {theta}, phi: {phi}")
    plt.xlabel("depth")
    plt.ylabel("sigma")
    plt.xlim(0.95, 1)
    plt.ylim(0, 0.004)
    os.makedirs(
        f"{HOMEDIR}/target_selection/figure/{b}/depth_contour/",
        exist_ok=True,
    )
    plt.savefig(
        f"{HOMEDIR}/target_selection/figure/{b}/depth_contour/b_{b}_{theta}_{phi}.png",
    )
    plt.close()
    # 線形補間関数を作成
    interp_func = interp1d(depth, sigma, kind="linear")
    pdb.set_trace()
    # df.depthとdf.sigmaを使って、等高線の下にある点を取得
    df["is_below_line"] = (df.sigma <= interp_func(df.depth)).values
    below_df = df[df["is_below_line"] == True]
    # below_df.to_csv("below_p_3sigma_TOIs_b_0.9.csv", index=False)


def plot_target_area_wrapper(arg):
    return plot_target_area(*arg)


def get_chisq(file: str):
    with open(file, mode="r") as f:
        chisquare = f.readlines()[7]
        chisquare = chisquare.split("    chi-square         = ")[-1]

    return float(chisquare)


def calc_scaling(ref_chisquare, depth, sigma, REF_DEPTH, REF_SIGMA):
    depth_scaling = ((1 - depth) / (1 - REF_DEPTH)) ** 2.5
    sigma_scaling = (sigma / REF_SIGMA) ** (-2)
    scaled_chisq = ref_chisquare * depth_scaling * sigma_scaling

    return scaled_chisq


REF_DEPTH = 0.99
REF_SIGMA = 0.0001
RINGFIT_CHISQ = 500
RINGFIT_BINS = 500
RINGFIT_NVARYS = 9
NORINGFIT_NVARYS = 6
HOMEDIR = os.getcwd()
df = pd.read_csv(f"{HOMEDIR}/exofop_tess_tois_20230526_sigma.csv")
df = pd.read_csv(f"{HOMEDIR}/below_p_3sigma_TOIs_b_0.9.csv")
depth_list = np.linspace(0.95, 0.999, 100)
sigma_list = np.linspace(0.00001, 0.0041, 100)
theta_list = [3, 10, 15, 20, 25, 30, 35, 40, 45, 50]
phi_list = [0, 10, 15, 20, 25, 30, 35, 40, 45, 50]
r_out_list = os.listdir(f"{HOMEDIR}/target_selection/data/b_0")
if ".DS_Store" in r_out_list:
    r_out_list.remove(".DS_Store")
    r_out_list.sort()

# if __name__ == "__main__":
"""theta, phi固定の場合"""
b_list = [
    0.6,
    0.8,
    1,
    0.1,
    0.3,
    0.5,
    0.7,
    0.9,
]
if ".DS_Store" in b_list:
    b_list.remove(".DS_Store")
    b_list.sort()
for b in tqdm(b_list):
    b_value = float(b)
    # dfの"b"の列から、値がb-0.05からb+0.049の行を取得
    limited_df = df[
        (df["b"] >= b_value - 0.050) & (df["b"] <= b_value + 0.049)
    ]
    b = f"b_{b}"

    plot_target_area(r_out_list, b, 45, 45, limited_df)
sys.exit()

"""depth, sigma固定の場合"""
for depth in tqdm(depth_list):
    for sigma in sigma_list:
        for b in b_list:
            # depth_sigma = os.listdir(f"/mwork2/umetanitb/research_umetani/depth_error/figure/{b}/")
            # depth_sigma.remove("depth_contour")
            # depth_sigma.remove("deg_contour")
            # theta_list = [int(file.split("_")[0].split("deg")[0]) for file in depth_sigma]
            # phi_list = [int(file.split("_")[-1].split("deg")[0]) for file in depth_sigma]
            if os.path.exists(
                f"{HOMEDIR}/depth_error/pvalues/pvalues_{depth}_{sigma}_{b}.csv"
            ):
                pass
            else:
                make_p_list_csv(theta_list, phi_list, b, depth, sigma)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plot_contour(ax, phi_list, theta_list, b, depth, sigma)
sys.exit()
min_flux_list = [
    0.99,
    0.98,
    0.97,
    0.96,
    0.95,
    0.999,
    0.998,
    0.997,
    0.996,
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
]
bin_error_list = np.arange(0.0001, 0.0041, 0.0001)
bin_error_list = np.around(bin_error_list, decimals=4)
b_list = os.listdir(f"{HOMEDIR}/depth_error/figure")
if ".DS_Store" in b_list:
    b_list.remove(".DS_Store")
for depth in tqdm(min_flux_list):
    for sigma in bin_error_list:
        for b in b_list:
            # print(f"{depth} {sigma} {b}")
            # theta, phiの各フォルダから同名のファイルを読み込む
            filelist = glob.glob(
                f"{HOMEDIR}/depth_error/data/{b}/*/{depth}_{sigma}.txt"
            )
            fig = plt.figure()
            ax = fig.add_subplot(111)

            # 軸範囲を設定
            ax.set_xlim(0, 50)
            ax.set_ylim(0, 50)
            for file in filelist:
                thetaphi = file.split("/")[-2]
                theta = float(thetaphi.split("_")[0].split("deg")[0])
                phi = float(thetaphi.split("_")[-1].split("deg")[0])
                with open(file, "r") as f:
                    p_value_line = f.readlines()[-1]
                    p_value = p_value_line.split(": ")[-1]
                    p_value = float(p_value)
                # plot area
                if p_value <= 0.01:
                    ax.scatter(theta, phi, fc="r", s=150, zorder=2)
                elif p_value <= 0.05:
                    ax.scatter(theta, phi, fc="orange", s=150, zorder=1)
            ax.set_title(f"b, depth, sigma: {b[2:]}, {depth}, {sigma}")
            plt.xlabel("theta")
            plt.ylabel("phi")
            os.makedirs(
                f"{HOMEDIR}/depth_error/figure/{b}/theta_phi_axis/",
                exist_ok=True,
            )
            plt.savefig(
                f"{HOMEDIR}/depth_error/figure/{b}/theta_phi_axis/pvalue_area_{depth}_{sigma}.png"
            )
            plt.close()

'''
for degree in degrees_paturn:
    print(degree, b)
    data = os.listdir(os.path.join(os.getcwd(), "data", b, degree))
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 軸範囲を設定
    ax.set_xlim(0.85, 1.0)
    ax.set_ylim(0, 0.004)
    for data in data:
        # get rp_rs and bin error from data
        rp_rs = float(data.split("_")[1])
        bin_error = float(data.split("_")[-1].split(".txt")[0])
        # get p_value
        with open(os.path.join(os.getcwd(), "data", b, degree, data), "r") as f:
            p_value_line = f.readlines()[-1]
            p_value = p_value_line.split(": ")[-1]
            p_value = float(p_value)

        # plot area
        if p_value <= 0.01:
            sigma3 = patches.Rectangle(
                xy=(0.0, 0),
                width=rp_rs,
                height=bin_error,
                fc="r",
                zorder=2,
            )
            ax.add_patch(sigma3)
            # ax.axvspan(
            #    rp_rs, 0.25, 0, bin_error / 0.002, color="red", zorder=2
            # )
        elif p_value <= 0.05:
            sigma2 = patches.Rectangle(
                xy=(0.90, 0),
                width=rp_rs - 0.90,
                height=bin_error,
                fc="orange",
                zorder=1,
            )
            ax.add_patch(sigma2)
            # ax.axvspan(
            #    rp_rs, 0.25, 0, bin_error / 0.002, color="coral", zorder=1
            # )
    # chose toi in [TOI101.01, TOI102.01, TOI105.01, TOI106.01, TOI107.01, TOI112.01, TOI114.01, TOI116.01,129.01, TOI135.01, TOI143.01, TOI150.01, TOI157.01, TOI159.01, TOI163.01]
    """
    df = df[
        (df["toi"] == "TOI112.01")
        | (df["toi"] == "TOI114.01")
        | (df["toi"] == "TOI116.01")
        | (df["toi"] == "TOI129.01")
        | (df["toi"] == "TOI143.01")
        | (df["toi"] == "TOI150.01")
        | (df["toi"] == "TOI157.01")
        | (df["toi"] == "TOI159.01")
        | (df["toi"] == "TOI163.01")
    ]
    markers1 = [
        ".",
        ",",
        "o",
        "v",
        "^",
        "<",
        ">",
        "1",
        "2",
        "3",
        ".",
        ",",
        "o",
        "v",
        ",",
    ]
    """
    for i, (min_flux, mean_flux_err) in enumerate(
        zip(df.depth, df.sigma)
    ):
        plt.scatter(
            min_flux,
            mean_flux_err,
            color="black",
            zorder=3,
            s=15,
            #marker=markers1[i],
        )
    """
    p_df = pd.read_csv(
        "{HOMEDIR}/calc_depsig.csv"
    )

    for i, (min_flux, mean_flux_err) in enumerate(
        zip(p_df.depth_list, p_df.sigma_list)
    ):
        plt.scatter(
            min_flux,
            mean_flux_err,
            color="red",
            zorder=4,
            s=15,
            marker=markers1[i],
        )
    """

    ax.set_title(f"theta_phi: {degree}")
    plt.xlabel("min flux")
    plt.ylabel("sigma")
    plt.savefig(
        f"{HOMEDIR}/depth_error/figure/{b}/pvalue_area_{degree}.png"
    )
    plt.close()
'''
