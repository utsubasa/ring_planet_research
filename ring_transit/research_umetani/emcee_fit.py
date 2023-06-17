# -*- coding: utf-8 -*-
import concurrent.futures
import os
import pdb
import warnings
from multiprocessing import Pool

import corner
import emcee
import lightkurve as lk
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import ascii
from numpy.linalg import svd

import c_compile_ring
from fit_model import model
from ring_planet import binning_lc

warnings.filterwarnings("ignore")


def ring_model(t, pdic, mcmc_pvalues=None):
    # Ring model
    # Input "x" (1d array), "pdic" (dic)
    # Ouput flux (1d array)
    if mcmc_pvalues is None:
        pass
    else:
        for i, param in enumerate(mcmc_pvalues):
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


def lnlike(mcmc_pvalues, t, y, yerr, pdic):
    return -0.5 * np.sum(((y - ring_model(t, pdic, mcmc_pvalues)) / yerr) ** 2)


def log_prior(mcmc_pvalues, mcmc_params, df_for_mcmc):
    # p_dict = dict(zip(mcmc_params, p))
    # rp_rs, theta, phi, r_in = p
    for i, param in enumerate(mcmc_params):
        if (
            df_for_mcmc["mins"][param]
            <= mcmc_pvalues[i]
            <= df_for_mcmc["maxes"][param]
        ):
            pass
        else:
            return -np.inf
        if param == "r_out" and mcmc_pvalues[i] > 3.0:
            return -np.inf
        else:
            pass
    # if 0.0 < theta < np.pi/2 and 0.0 < phi < np.pi/2 and 0.0 < rp_rs < 1 and 1.0 < r_in < 7.0:
    #    return 0.0
    return 0.0


def lnprob(mcmc_pvalues, t, y, yerr, mcmc_params, pdic, df_for_mcmc):
    lp = log_prior(mcmc_pvalues, mcmc_params, df_for_mcmc)
    if not np.isfinite(lp):
        return -np.inf
    chi_square = np.sum(((y - ring_model(t, pdic, mcmc_pvalues)) / yerr) ** 2)
    print(chi_square)

    return lp + lnlike(mcmc_pvalues, t, y, yerr, pdic)


def plot_ring(
    rp_rs,
    rin_rp,
    rout_rin,
    b,
    theta,
    phi,
    file_name,
    chi_square,
    flux_data,
    mcmc_df,
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
    ax.set_title(
        f"chisq={chi_square}, dof={len(flux_data)-len(mcmc_df.index)}"
    )
    plt.axis("scaled")
    ax.set_aspect("equal")
    # os.makedirs(f'./lmfit_result/illustration/{TOInumber}', exist_ok=True)
    # plt.savefig(f'./lmfit_result/illustration/{TOInumber}/{file_name}', bbox_inches="tight")
    plt.savefig(file_name, bbox_inches="tight")
    plt.close()


def run_sampler(
    iterations,
    pos,
    t,
    flux_data,
    flux_err_data,
    mcmc_params,
    pdic,
    df_for_mcmc,
):
    nwalkers, ndim = pos.shape
    autocorr = np.empty(iterations)
    index = 0
    old_tau = np.inf

    def check_convergence(sampler, tau, old_tau):
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        return converged

    def sample_step(sample):
        nonlocal index, old_tau
        if sampler.iteration % 100:
            return
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1
        if check_convergence(sampler, tau, old_tau):
            sampler.reset()
            return True
        old_tau = tau

    with concurrent.futures.ProcessPoolExecutor(4) as executor:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            lnprob,
            args=(
                t,
                flux_data,
                flux_err_data.mean(),
                mcmc_params,
                pdic,
                df_for_mcmc,
            ),
            pool=executor,
        )
        for _ in sampler.sample(pos, iterations=iterations, progress=True):
            if sample_step(_):
                break
    return sampler, ndim


def step_plotter(ndim, sampler, mcmc_params, try_n, savefig_dir):
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = mcmc_params
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    plt.savefig(f"{savefig_dir}/step_{try_n}.png")
    plt.close()


def corner_plotter(flat_samples, labels, try_n, savefig_dir):
    """
    truths = []
    for param in labels:
        truths.append(pdic_saturnlike[param])
    fig = corner.corner(flat_samples, labels=labels, truths=truths);
    """
    _ = corner.corner(flat_samples, labels=labels)
    plt.savefig(f"{savefig_dir}/corner_{try_n}.png")
    plt.close()


def lc_plotter(
    t,
    flux_data,
    flux_err_data,
    flat_samples,
    pdic,
    try_n,
    inds,
    savefig_dir,
):
    plt.errorbar(
        t,
        flux_data,
        yerr=flux_err_data,
        fmt=".k",
        capsize=0,
        alpha=0.1,
    )
    for ind in inds:
        sample = flat_samples[ind]
        flux_model = ring_model(t, pdic, sample)
        plt.plot(t, flux_model, "C1", alpha=0.5)
        # fit_report = lmfit.fit_report(ring_res)
        # print(fit_report)
    # plt.plot(t, ymodel, "k", label="truth")
    plt.legend(fontsize=14)
    # plt.xlim(0, 10)
    plt.xlabel("orbital phase")
    plt.ylabel("flux")
    plt.title(f"n_bins: {len(flux_data)}")
    plt.savefig(f"{savefig_dir}/fit_result_{try_n}.png")
    plt.close()


def main():
    # mcmcで使用するパラメータを定義
    params_df = pd.DataFrame(
        params_data, columns=["mins", "maxes", "vary_flags"], index=p_names
    )
    df_for_mcmc = params_df[params_df["vary_flags"]]

    # 各天体のパラメータを参照するdfを定義
    df = pd.read_csv(f"{HOMEDIR}/exofop_tess_tois_20230526_sigma.csv")
    df["TOI"] = df["TOI"].astype(str)

    # ターゲットとなるTOIでforループ
    below_df = pd.read_csv(f"{HOMEDIR}/below_p_3sigma_TOIs.csv")
    TOIlist = below_df["TOI"].values
    for toi in TOIlist:
        TOInumber = "TOI" + str(toi)
        savefig_dir = f"{HOMEDIR}/mcmc_result/figure/{TOInumber}"
        savedata_dir = f"{HOMEDIR}/mcmc_result/data/{TOInumber}"
        csvfile = f"{HOMEDIR}/folded_lc_data/0605_p0.05/{TOInumber}.csv"
        init_params_dir = f"{HOMEDIR}/lmfit_res/sap_0605_p0.05/fit_p_data/ring_model/{TOInumber}"
        param_df = df[df["TOI"] == TOInumber[3:]]
        # durationは算出したものを使う（ringfitのときと同じにする）
        load_dur_per_file = f"{HOMEDIR}/folded_lc_data/fit_report_0605_p0.05/{TOInumber}_folded.txt"

        # durationの読み込み。読み込めない場合はexofopの値を使う
        try:
            with open(
                load_dur_per_file,
                "r",
            ) as f:
                # durationline = f.readlines()[-5:].split(' ')[-1]
                durationline, _, _, periodline, _ = f.readlines()[-5:]
                durationline = durationline.split(" ")[-1]
                duration = np.float(durationline)
        except FileNotFoundError:
            with open("./folded.txt_FileNotFoundError.txt", "a") as f:
                f.write(TOInumber + "\n")
            try:
                duration = param_df["Duration (hours)"].values[0] / 24
            except IndexError:
                with open("./noparameterindf.txt", "a") as f:
                    f.write(TOInumber + "\n")
                continue

        # lightcurveを読み込む
        try:
            folded_table = ascii.read(csvfile)
        except FileNotFoundError:
            continue
        folded_lc = lk.LightCurve(data=folded_table)
        folded_lc = folded_lc[
            (folded_lc.time.value < duration * 0.7)
            & (folded_lc.time.value > -duration * 0.7)
        ]

        # 500binにビニングする
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

        # p_csv、mcmcの初期値となるパラメータを読み込む
        p_csvs = os.listdir(init_params_dir)
        # p_csvsのそれぞれの要素を"_"で区切った時の2番めの値(F_obs)が最も大きいものを選ぶ
        p_csvs = sorted(
            p_csvs, key=lambda x: float(x.split("_")[1]), reverse=True
        )
        p_csv = p_csvs[0]

        mcmc_df = pd.read_csv(f"{init_params_dir}/{p_csv}")
        mcmc_df.index = p_names
        mcmc_df.at["phi", "input_value"] = 45 / 180 * np.pi
        mcmc_df.at["theta", "input_value"] = 45 / 180 * np.pi
        pdic = mcmc_df["input_value"].to_dict()

        # mcmcで事後分布推定しないパラメータをここで設定
        mcmc_df.at["t0", "vary_flags"] = False
        mcmc_df.at["phi", "vary_flags"] = False
        mcmc_df.at["theta", "vary_flags"] = False
        mcmc_df = mcmc_df[mcmc_df["vary_flags"] == True]
        mcmc_params = mcmc_df.index.tolist()
        for try_n in range(5):
            mcmc_pvalues = mcmc_df["output_value"].values
            print("mcmc_params: ", mcmc_params)
            print("mcmc_pvalues: ", mcmc_pvalues)
            pos = mcmc_pvalues + 1e-2 * np.random.randn(32, len(mcmc_pvalues))
            # pos = np.array([rp_rs, theta, phi, r_in, r_out]) + 1e-8 * np.random.randn(32, 5)
            sampler, ndim = run_sampler(
                MAX_N,
                pos,
                t,
                flux_data,
                flux_err_data,
                mcmc_params,
                pdic,
                df_for_mcmc,
            )
            """
            nwalkers, ndim = pos.shape
            # filename = "emcee_{0}.h5".format(datetime.datetime.now().strftime('%y%m%d%H%M'))
            # backend = emcee.backends.HDFBackend(filename)
            # backend.reset(nwalkers, ndim)
            autocorr = np.empty(max_n)
            old_tau = np.inf

            ###mcmc run###
            # sampler.run_mcmc(pos, max_n, progress=True)
            if __name__ == "__main__":
                with Pool(processes=4) as pool:
                    sampler = emcee.EnsembleSampler(
                        nwalkers,
                        ndim,
                        lnprob,
                        args=(t, flux_data, flux_err_data.mean(), mcmc_params),
                        pool=pool,
                    )
                    # pos = sampler.run_mcmc(pos, max_n)
                    # sampler.reset()
                    for sample in sampler.sample(
                        pos, iterations=max_n, progress=True
                    ):
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
                plt.plot(n, n / 100.0, "--k")
                plt.plot(n, y)
                plt.xlim(0, n.max())
                plt.ylim(0, y.max() + 0.1 * (y.max() - y.min()))
                plt.xlabel("number of steps")
                plt.ylabel(r"mean $\hat{\tau}$")
                plt.savefig(f'tau_{try_n}.png')\
                ##plt.show()
                plt.close()
                print(tau)
                
                """
            os.makedirs(savefig_dir, exist_ok=True)
            # step visualization
            step_plotter(ndim, sampler, mcmc_params, try_n, savefig_dir)

            # corner visualization
            # samples = sampler.flatchain
            flat_samples = sampler.get_chain(
                discard=DISCARD, thin=15, flat=True
            )
            corner_plotter(ndim, sampler, mcmc_params, try_n)
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
            # ライトカーブとモデルのプロット
            lc_plotter(
                t,
                flux_data,
                flux_err_data,
                flat_samples,
                pdic,
                try_n,
                inds,
            )
            os.makedirs(f"{savefig_dir}/illustration", exist_ok=True)
            # ポンチ絵の作成とcsvへの保存
            for ind in inds:
                sample = flat_samples[ind]
                flux_model = ring_model(t, pdic, sample)
                ###csvに書き出し###
                mcmc_res_df = mcmc_df
                mcmc_res_df["output_value"] = sample
                rp_rs = mcmc_res_df.at["rp_rs", "output_value"]
                rin_rp = mcmc_res_df.at["r_in", "output_value"]
                rout_rin = mcmc_res_df.at["r_out", "output_value"]
                b = mcmc_res_df.at["b", "output_value"]
                theta = mcmc_res_df.at["theta", "output_value"]
                phi = mcmc_res_df.at["phi", "output_value"]
                chi_square = np.sum(
                    ((flux_model - flux_data) / flux_err_data) ** 2
                )
                file_name = (
                    f"{savefig_dir}/illustration/{chi_square:.0f}_{try_n}.pdf"
                )
                os.makedirs(savedata_dir, exist_ok=True)
                mcmc_res_df.to_csv(
                    f"{savedata_dir}_{chi_square:.0f}_{try_n}.csv",
                    header=True,
                    index=False,
                )
                plot_ring(
                    rp_rs,
                    rin_rp,
                    rout_rin,
                    b,
                    theta,
                    phi,
                    file_name,
                    chi_square,
                    flux_data,
                    mcmc_df,
                )


p_names = [
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

params_data = [
    [0.0, 1.0, True],
    [0.0, 1.0, True],
    [-0.1, 0.1, False],
    [0.0, 100.0, False],
    [0.003, 0.5, True],
    [1.0, 100.0, True],
    [0.0, 1.2, True],
    [0.9, 1.1, False],
    [0.0, np.pi, False],
    [0.0, np.pi, False],
    [0.0, 1.0, False],
    [1.0, 3.0, False],
    [1.1, 1.7, True],
    [-0.1, 0.1, False],
    [-0.1, 0.1, False],
    [0.0, 0.0, False],
    [0.0, 0.0, False],
]

HOMEDIR = os.getcwd()
MAX_N = 10000
DISCARD = 2500
INDEX = 0

if __name__ == "__main__":
    main()
