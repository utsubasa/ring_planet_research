import os
import pdb
import pickle
import sys
import time
import warnings
from concurrent import futures
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Sequence, Tuple

import batman
import lightkurve as lk
import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from astropy.io import ascii
from astropy.table import vstack

import ring_planet

warnings.filterwarnings("ignore")

"""定数の定義"""
with open(os.getcwd() + "/no_data_found_toi.txt", "rb") as f:
    no_data_found_list = pickle.load(f)  # short がないか、SPOCがないか
# todo: 除外する系をtableの中に記載する
no_perioddata_list = [
    1134.01,
    1897.01,
    2423.01,
    2666.01,
    4465.01,
]  # exofopの表にperiodの記載無し。1567.01,1656.01もperiodなかったがこちらはcadence=’short’のデータなし。
no_signal_list = [2218.01]  # トランジットのsignalが無いか、ノイズに埋もれて見えない
multiplanet_list = [1670.01, 201.01, 822.01]  # , 1130.01]
startrend_list = [4381.01, 1135.01, 1025.01, 1830.01, 2119.01]
flare_list = [212.01, 1779.01, 2119.01]
two_epoch_list = [
    671.01,
    1963.01,
    1283.01,
    758.01,
    1478.01,
    3501.01,
    845.01,
    121.01,
    1104.01,
    811.01,
    3492.01,
    1312.01,
    1861.01,
    665.01,
    224.01,
    2047.01,
    5379.01,
    5149.01,
    5518.01,
    5640.01,
    319.01,
    2783.01,
    5540.01,
    1840.01,
    5686.01,
    5226.01,
    937.01,
    4725.01,
    4731.01,
    5148.01,
    1130.01,
]
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
    1344.01,
]
duration_ng = [
    129.01,
    182.01,
    1059.01,
    1182.01,
    1425.01,
    1455.01,
    1811.01,
    2154.01,
    3910.01,
]
trend_ng = [
    1069.01,
    1092.01,
    1141.01,
    1163.01,
    1198.01,
    1270.01,
    1299.01,
    1385.01,
    1454.01,
    1455.01,
    1647.01,
    1796.01,
]
fold_ng = [986.01]

HOMEDIR = os.getcwd() + "/SAP_fitting_result_under_p3sigma_20230530"
# 既に前処理したTOIの重複した前処理を回避するためのTOIのリスト
try:
    donelist = os.listdir(f"{HOMEDIR}/folded_lc/obs_t0/data")
    donelist = [s for s in donelist if "TOI" in s]
    donelist = [s.lstrip("TOI") for s in donelist]
    donelist = [float(s.strip(".csv")) for s in donelist]
    donelist = [float(s) for s in donelist]
    # str型に変換
    donelist = [str(s) for s in donelist]
except FileNotFoundError:
    donelist = []

oridf = pd.read_csv(os.getcwd() + "/exofop_tess_tois_20230526_sigma.csv")
# df = oridf[oridf["Planet SNR"] > 100]
df = oridf
df = df[~(df["TESS Disposition"] == "EB")]
df = df[~(df["TFOPWG Disposition"] == "FP")]
df = df.sort_values("Planet SNR", ascending=False)

"""処理を行わないTOIを選択する"""
df = df.set_index(["TOI"])
exclude_list = (
    no_data_found_list
    + two_epoch_list
    + no_perioddata_list
    + no_signal_list
    + multiplanet_list
    + startrend_list
    + flare_list
    + ignore_list
    + duration_ng
    + fold_ng
    + trend_ng
    + donelist
)
exclude_list = list(set(exclude_list))
df = df[~df.index.isin(exclude_list)]
df = df.reset_index()
df["TOI"] = df["TOI"].astype(str)
TOIlist = df["TOI"]
below_df = pd.read_csv(
    "/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/below_p_0.05_TOIs.csv"
)
TOIlist = below_df["TOI"]
# str
TOIlist = [str(s) for s in TOIlist]
# TOIlistから既に処理したTOIを除外する
TOIlist = [s for s in TOIlist if s not in donelist]


def tpfcorrect(
    tpf,
    AFTER_PLD_FIG_DIR,
    AFTER_PLD_DATA_DIR,
    TOInumber,
    period,
    duration,
    transit_time,
):
    print(f"correcting: {tpf}")
    aperture_mask = tpf.create_threshold_mask(3)
    lc = tpf.to_lightcurve()
    try:
        pld = tpf.to_corrector("pld")
    except ValueError:
        nan_mask = np.isnan(lc.flux)
        lc = lc[~nan_mask]
        tpf = tpf[~nan_mask]
        nan_mask = np.isnan(lc.flux_err)
        lc = lc[~nan_mask]
        tpf = tpf[~nan_mask]
        pld = tpf.to_corrector("pld")
        # この処理を行ったTOIとtpfの名前を記録する
        with open("pld_error_toi.dat", "a") as f:
            f.write(f"{TOInumber}: {tpf}\n")
    try:
        corrected_lc = pld.correct()
    except ValueError:
        pdb.set_trace()

    # masking transit signal
    transit_mask = corrected_lc.create_transit_mask(
        period=period, duration=duration, transit_time=transit_time
    )
    corrected_lc = pld.correct(cadence_mask=~transit_mask, restore_trend=False)

    # 補正前後のlcをプロットする
    ax = lc.remove_outliers(sigma=10).errorbar(label="SAP", zorder=1)
    corrected_lc.errorbar(
        ax=ax,
        column="flux",
        color="red",
        label="after PLD",
    )
    corrected_lc[transit_mask].scatter(ax=ax, c="green", label="transit_mask")
    ax.legend()
    ax.set_title(f"TOI {TOI}")
    plt.savefig(f"{AFTER_PLD_FIG_DIR}/{TOInumber}_sector{tpf.sector}.png")
    plt.close()

    corrected_lc = ring_planet.correct_sap_lc(corrected_lc)

    # save corrected lc
    corrected_lc.to_csv(
        f"{AFTER_PLD_DATA_DIR}/{TOInumber}_sector{tpf.sector}.csv",
        overwrite=True,
    )


def preprocess(TOI, df):
    TOI = str(TOI)
    TOInumber = "TOI" + TOI
    print("analysing: ", "TOI" + str(TOI))

    # 惑星、主星の各パラメータを取得
    param_df = df[df["TOI"] == TOI]
    try:
        param_dic = ring_planet.get_params_from_table(param_df, TOI)
    except ValueError:
        with open("no_data_found_toi_0523.txt", "a") as f:
            f.write(f"{TOInumber}\n")
        return
    duration = param_dic["duration"]
    period = param_dic["period"]
    transit_time = param_dic["transit_time"]
    transit_time_error = param_dic["transit_time_error"]
    rp_rs = param_dic["rp_rs"]

    # もしもduration, period, transit_timeどれかのパラメータがnanだったらそのTOIを記録して、処理はスキップする
    if np.sum(np.isnan([duration, period, transit_time])) != 0:
        with open("nan3params_toi.dat", "a") as f:
            f.write(
                f"{TOInumber}: {np.isnan([duration, period, transit_time])}\n"
            )
        return

    # 保存場所のセッティング　# todo: 各保存先の確認,
    EACH_LC_DIR = f"{HOMEDIR}/each_lc"
    FOLD_LC_DIR = f"{HOMEDIR}/folded_lc"
    AFTER_PLD_FIG_DIR = f"{HOMEDIR}/after_pld/figure/{TOInumber}"
    AFTER_PLD_DATA_DIR = f"{HOMEDIR}/after_pld/data/{TOInumber}"
    CAL_UNDER90PER_DIR = f"{EACH_LC_DIR}/under_90%/calc_t0/{TOInumber}"
    OBS_UNDER90PER_DIR = CAL_UNDER90PER_DIR.replace("calc_t0", "obs_t0")
    CURVEFIT_DIR = f"{EACH_LC_DIR}/curvefit/{TOInumber}"
    TRANSITFIT_DIR = f"{EACH_LC_DIR}/transit_fit/{TOInumber}"
    TRANSIT_POLYFIT_DIR = f"{EACH_LC_DIR}/transit_and_polyfit/{TOInumber}"
    CAL_PROCESS_FIG_DIR = (
        f"{EACH_LC_DIR}/after_process/calc_t0/figure/{TOInumber}"
    )
    CAL_PROCESS_DATA_DIR = (
        f"{EACH_LC_DIR}/after_process/calc_t0/data/{TOInumber}"
    )
    OBS_PROCESS_FIG_DIR = CAL_PROCESS_FIG_DIR.replace("calc_t0", "obs_t0")
    OBS_PROCESS_DATA_DIR = CAL_PROCESS_DATA_DIR.replace("calc_t0", "obs_t0")
    CAL_MODELFIT_DIR = f"{EACH_LC_DIR}/modelresult/calc_t0/{TOInumber}"
    OBS_MODELFIT_DIR = CAL_MODELFIT_DIR.replace("calc_t0", "obs_t0")
    CAL_FOLDMODELFIT_DIR = f"{FOLD_LC_DIR}/modelresult/calc_t0"
    OBS_FOLDMODELFIT_DIR = CAL_FOLDMODELFIT_DIR.replace("calc_t0", "obs_t0")
    CAL_FOLD_FIG_DIR = f"{FOLD_LC_DIR}/calc_t0/figure"
    OBS_FOLD_FIG_DIR = CAL_FOLD_FIG_DIR.replace("calc_t0", "obs_t0")
    CAL_FOLD_DATA_DIR = f"{FOLD_LC_DIR}/calc_t0/data"
    OBS_FOLD_DATA_DIR = CAL_FOLD_DATA_DIR.replace("calc_t0", "obs_t0")
    SAVE_OC_DIAGRAM = f"{HOMEDIR}/calc_obs_transit_time"
    SAVE_ESTIMATED_PER = f"{HOMEDIR}/estimate_period"
    T0DIR = f"{HOMEDIR}/t0list"

    # 保存場所のディレクトリがなければ作成する
    dirlist = [
        EACH_LC_DIR,
        FOLD_LC_DIR,
        AFTER_PLD_FIG_DIR,
        AFTER_PLD_DATA_DIR,
        CAL_UNDER90PER_DIR,
        OBS_UNDER90PER_DIR,
        CURVEFIT_DIR,
        TRANSITFIT_DIR,
        TRANSIT_POLYFIT_DIR,
        CAL_PROCESS_FIG_DIR,
        CAL_PROCESS_DATA_DIR,
        OBS_PROCESS_FIG_DIR,
        OBS_PROCESS_DATA_DIR,
        CAL_MODELFIT_DIR,
        OBS_MODELFIT_DIR,
        CAL_FOLDMODELFIT_DIR,
        OBS_FOLDMODELFIT_DIR,
        CAL_FOLD_FIG_DIR,
        OBS_FOLD_FIG_DIR,
        CAL_FOLD_DATA_DIR,
        OBS_FOLD_DATA_DIR,
        SAVE_OC_DIAGRAM,
        SAVE_ESTIMATED_PER,
        T0DIR,
    ]
    for dir in dirlist:
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
    """
    # lightkurveを用いてSPOCが作成した2min cadenceの全セクターのライトカーブをダウンロードする.
    search_result = lk.search_targetpixelfile(
        f"TOI {TOI[:-3]}",
        mission="TESS",
        cadence="short",
        author="SPOC",
        sector=param_df.Sectors.values[0].split(","),
    )
    tpf_collection = search_result.download_all()

    # tpfごとにpldを用いて補正を行い、それを保存する

    for tpf in tpf_collection:
        # f"{AFTER_PLD_DATA_DIR}/{TOInumber}_sector{tpf.sector}.csv"が存在している場合はスキップする

        if os.path.exists(
            f"{AFTER_PLD_DATA_DIR}/{TOInumber}_sector{tpf.sector}.csv"
        ):
            print(
                f"already exist: {AFTER_PLD_DATA_DIR}/{TOInumber}_sector{tpf.sector}.csv"
            )
            continue

        tpfcorrect(
            tpf,
            AFTER_PLD_FIG_DIR,
            AFTER_PLD_DATA_DIR,
            TOInumber,
            period,
            duration,
            transit_time,
        )
    # tpfごとに補正したlcを読み込み、それを合成する
    """
    ring_planet.get_lc_from_mast(TOI, AFTER_PLD_DATA_DIR, lc_type="sap")
    lc = ring_planet.stack_lc_from_csv(AFTER_PLD_DATA_DIR)
    lc = lc.remove_nans()
    # flux=0のデータを除外する
    lc = lc[lc.flux != 0]

    # transit_timeとperiodからターゲットの惑星のtransit time listを作成
    transit_time_list = ring_planet.set_transit_times(
        transit_time=transit_time,
        period=period,
        lc=lc,
    )

    # 多惑星系の場合、ターゲットのトランジットに影響があるかを判断するため、記録しておく。
    print("judging whether other planet transit is included in the data...")
    other_p_df = oridf[oridf["TIC ID"] == param_df["TIC ID"].values[0]]
    if len(other_p_df.index) != 1:
        with open("multiplanet_toi.dat", "a") as f:
            f.write(f"{TOInumber}\n")
        return

    # 1stloop: 各エポックでトランジットフィット、カーブフィットを別々に行いlightcurveの前処理を行う
    # 値を格納するリストの定義
    outliers = []
    t0list = []
    t0errlist = []
    num_list = []
    # ax = lc.scatter()
    for i, mid_transit_time in enumerate(transit_time_list):  # todo: 並列化
        print(f"preprocessing...epoch: {i}")
        # トランジットの中心時刻から±duration*2.5の時間帯を切り取る
        epoch_start = mid_transit_time - (duration * 2.5)
        epoch_end = mid_transit_time + (duration * 2.5)
        tmp = lc[lc.time.value > epoch_start]
        each_lc = tmp[tmp.time.value < epoch_end]
        each_lc = (
            each_lc.fold(period=period, epoch_time=mid_transit_time)
            .normalize()
            .remove_nans()
        )

        # 解析中断条件を満たさないかチェック
        data_survival_rate = ring_planet.calc_data_survival_rate(
            each_lc, duration
        )
        if data_survival_rate < 90:
            if data_survival_rate != 0.0:
                ax = each_lc.errorbar()
                ax.set_title(f"{data_survival_rate:2f}% useable")
                plt.savefig(f"{CAL_UNDER90PER_DIR}/{TOInumber}_{str(i)}.png")
                plt.close()
            t0list.append(mid_transit_time)
            t0errlist.append(np.nan)
            continue
        else:
            num_list.append(i)

        # 外れ値除去と多項式フィッティングを外れ値が検知されなくなるまで繰り返す
        while True:
            # curvefitを正確にするためtransitfitでt0を求めておく
            try:
                transit_res = ring_planet.transit_fitting(
                    each_lc, rp_rs, period
                )
            except ValueError:
                pdb.set_trace()
            curvefit_res = ring_planet.curve_fitting(
                each_lc, duration, transit_res
            )
            plot_lc = ring_planet.PlotCurvefit(
                savedir=CURVEFIT_DIR,
                savefile=f"{TOInumber}_{str(i)}.png",
                lc=each_lc,
                fit_res=curvefit_res,
            )
            plot_lc.plot()
            plot_lc.save()

            each_lc = ring_planet.curvefit_normalize(
                each_lc, curvefit_res.params
            )
            transit_res = ring_planet.transit_fitting(each_lc, rp_rs, period)
            transit_model = ring_planet.no_ring_transitfit(
                transit_res.params,
                each_lc.time.value,
                each_lc.flux.value,
                each_lc.flux_err.value,
                return_model=True,
            )
            outlier_bools = ring_planet.detect_outliers(each_lc, transit_model)
            inverse_mask = np.logical_not(outlier_bools)
            if np.all(inverse_mask):
                plot_lc = ring_planet.PlotLightcurveWithModel(
                    savedir=TRANSITFIT_DIR,
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
                plot_after1stloop_lc = ring_planet.PlotLightcurve(
                    savedir=CAL_PROCESS_FIG_DIR,
                    savefile=f"{TOInumber}_{str(i)}.png",
                    lc=each_lc,
                )
                plot_after1stloop_lc.plot_lightcurve()
                plot_after1stloop_lc.save()
                each_lc.write(
                    f"{CAL_PROCESS_DATA_DIR}/{TOInumber}_{str(i)}.csv",
                    overwrite=True,
                )
                # save_each_lc(each_lc, "calc_t0")
                # t0のズレを補正したmidtransittimelistを作るためt0list,t0errlistそれぞれappend
                t0list.append(
                    transit_res.params["t0"].value + mid_transit_time
                )
                t0errlist.append(transit_res.params["t0"].stderr)

                outliers = []

                break
            else:
                outliers.append(each_lc[outlier_bools])
                each_lc = ring_planet.clip_outliers(
                    each_lc,
                    outlier_bools,
                )

        with open(
            f"{CAL_MODELFIT_DIR}/{TOInumber}_{str(i)}.txt",
            "a",
        ) as f:
            print(lmfit.fit_report(transit_res), file=f)

    # plt.show()
    # pdb.set_trace()
    with open(f"{T0DIR}/{TOInumber}.pkl", "wb") as f:
        pickle.dump(t0list, f)
    # folded_lcに対してtransit fitting & remove outliers. transit parametersを得る
    print("folding and calculate duration...")
    folded_lc = ring_planet.stack_lc_from_csv(
        loaddir=CAL_PROCESS_DATA_DIR,
    )

    outliers = []
    while True:
        fold_res = ring_planet.transit_fitting(
            folded_lc,
            rp_rs,
            period,
            fitting_model=ring_planet.no_ring_transitfit,
        )
        transit_model = ring_planet.no_ring_transitfit(
            fold_res.params,
            folded_lc.time.value,
            folded_lc.flux.value,
            folded_lc.flux_err.value,
            return_model=True,
        )
        outlier_bools = ring_planet.detect_outliers(folded_lc, transit_model)
        inverse_mask = np.logical_not(outlier_bools)
        if np.all(inverse_mask):
            break

        else:
            outliers.append(folded_lc[outlier_bools])
            folded_lc = ring_planet.clip_outliers(
                folded_lc,
                outlier_bools,
            )

    # 各エポックでのtransit fittingで得たmid_transit_timeのリストからorbital period、durationを算出
    period, period_err = ring_planet.estimate_period(
        t0list,
        t0errlist,
        num_list,
        transit_time_list,
        TOInumber,
        SAVE_ESTIMATED_PER,
        period,
    )
    a_rs = fold_res.params["a"].value
    b = fold_res.params["b"].value
    inc = np.arccos(b / a_rs)
    if np.isnan(rp_rs):
        rp_rs = fold_res.params["rp"].value
    duration = (period / np.pi) * np.arcsin(
        (1 / a_rs)
        * (np.sqrt(np.square(1 + rp_rs) - np.square(b)) / np.sin(inc))
    )
    plot_lc = ring_planet.PlotLightcurveWithModel(
        savedir=CAL_FOLD_FIG_DIR,
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

    folded_lc.write(f"{CAL_FOLD_DATA_DIR}/{TOInumber}.csv", overwrite=True)
    obs_t0_idx = np.abs(np.asarray(t0list) - transit_time).argmin()
    with open(
        f"{CAL_FOLDMODELFIT_DIR}/{TOInumber}_folded.txt",
        "a",
    ) as f:
        print(lmfit.fit_report(fold_res), file=f)
        print(f"calculated duration[day]: {duration}", file=f)
        print(f"obs_transit_time[day]: {t0list[obs_t0_idx]}", file=f)
        print(f"obs_transit_time_err[day]: {t0errlist[obs_t0_idx]}", file=f)
        print(f"obs_period[day]: {period}", file=f)
        print(f"obs_period_err[day]: {period_err}", file=f)

    transit_time_list = np.array(transit_time_list)
    ring_planet.make_oc_diagram(
        t0list,
        t0errlist,
        transit_time_list,
        transit_time_error,
        TOInumber,
        SAVE_OC_DIAGRAM,
    )
    """fittingで得たtransit time listを読み込む"""
    with open(f"{T0DIR}/{TOInumber}.pkl", "rb") as f:
        t0list = pickle.load(f)
    obs_t0_list = t0list
    outliers = []

    """transit parametersをfixして、baseline,t0を決める"""
    for i, mid_transit_time in enumerate(obs_t0_list):
        print(f"reprocessing...epoch: {i}")
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
        data_survival_rate = ring_planet.calc_data_survival_rate(
            each_lc, duration
        )
        if data_survival_rate < 90:
            if data_survival_rate != 0.0:
                ax = each_lc.errorbar()
                ax.set_title(f"{data_survival_rate:4f} useable")
                os.makedirs(OBS_UNDER90PER_DIR, exist_ok=True)
                plt.savefig(f"{OBS_UNDER90PER_DIR}/{TOInumber}_{str(i)}.png")
                plt.close()
            continue

        while True:
            # curvefitを正確に行うために1回transitfitしている
            transit_res = ring_planet.transit_fitting(each_lc, rp_rs, period)
            # 初期値を得るためのcurvefit
            curvefit_res = ring_planet.curve_fitting(
                each_lc, duration, transit_res
            )

            res = ring_planet.transit_fitting(
                each_lc,
                rp_rs,
                period,
                fitting_model=ring_planet.no_ring_transit_and_polynomialfit,
                transitfit_params=fold_res.params,
                curvefit_params=curvefit_res.params,
            )
            (
                flux_model,
                transit_model,
                polynomial_model,
            ) = ring_planet.no_ring_transit_and_polynomialfit(
                res.params,
                each_lc.time.value,
                each_lc.flux.value,
                each_lc.flux_err.value,
                return_model=True,
            )
            outlier_bools = ring_planet.detect_outliers(each_lc, flux_model)
            inverse_mask = np.logical_not(outlier_bools)
            if np.all(inverse_mask):
                # each_lc.time = each_lc.time - res.params["t0"].value
                plot_lc = ring_planet.PlotLightcurveWithModel(
                    savedir=TRANSIT_POLYFIT_DIR,
                    savefile=f"{TOInumber}_{str(i)}.png",
                    lc=each_lc,
                    outliers=outliers,
                    model={
                        "transit model": (transit_model, "blue"),
                        "polynomial model": (polynomial_model, "red"),
                        "fitting model": (flux_model, "black"),
                    },
                )
                plot_lc.plot_lightcurve()
                plot_lc.plot_model()
                plot_lc.plot_residuals()
                plot_lc.plot_outliers()
                plot_lc.configs()
                plot_lc.save()

                each_lc = ring_planet.curvefit_normalize(each_lc, res.params)
                each_lc.time = each_lc.time - res.params["t0"].value
                plot_after2ndloop_lc = ring_planet.PlotLightcurve(
                    savedir=OBS_PROCESS_FIG_DIR,
                    savefile=f"{TOInumber}_{str(i)}.png",
                    lc=each_lc,
                )
                plot_after2ndloop_lc.plot_lightcurve()
                plot_after2ndloop_lc.save()
                each_lc.write(
                    f"{OBS_PROCESS_DATA_DIR}/{TOInumber}_{str(i)}.csv",
                    overwrite=True,
                )
                # save_each_lc(each_lc, "obs_t0")
                outliers = []
                break
            else:
                outliers.append(each_lc[outlier_bools])
                each_lc = ring_planet.clip_outliers(
                    each_lc,
                    outlier_bools,
                )
        with open(
            f"{OBS_MODELFIT_DIR}/{TOInumber}_{str(i)}.txt",
            "a",
        ) as f:
            print(lmfit.fit_report(res), file=f)

        # ringfit(i, each_lc)

    """最終的なfolded_lcを生成する。"""
    print("refolding...")
    folded_lc = ring_planet.stack_lc_from_csv(
        loaddir=OBS_PROCESS_DATA_DIR,
    )
    outliers = []
    while True:
        fold_res = ring_planet.transit_fitting(
            folded_lc,
            rp_rs,
            period,
            fitting_model=ring_planet.no_ring_transitfit,
        )
        transit_model = ring_planet.no_ring_transitfit(
            fold_res.params,
            folded_lc.time.value,
            folded_lc.flux.value,
            folded_lc.flux_err.value,
            return_model=True,
        )
        outlier_bools = ring_planet.detect_outliers(folded_lc, transit_model)
        inverse_mask = np.logical_not(outlier_bools)
        if np.all(inverse_mask):
            # save result
            plot_lc = ring_planet.PlotLightcurveWithModel(
                savedir=OBS_FOLD_FIG_DIR,
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

            folded_lc.write(
                f"{OBS_FOLD_DATA_DIR}/{TOInumber}.csv",
                overwrite=True,
            )
            with open(
                f"{OBS_FOLDMODELFIT_DIR}/{TOInumber}_folded.txt",
                "a",
            ) as f:
                print(lmfit.fit_report(fold_res), file=f)
            break
        else:
            outliers.append(folded_lc[outlier_bools])
            folded_lc = ring_planet.clip_outliers(
                folded_lc,
                outlier_bools,
            )
    print(f"Analysis completed: {TOInumber}")


if __name__ == "__main__":
    for TOI in TOIlist:
        preprocess(TOI, df)
    """
    done_tois = np.loadtxt("TOI_a_b_depth_sigma.txt")[:, 0].tolist()
    nodata_tois = np.loadtxt("no_data_found_20230409.txt").tolist()
    noperiod_tois = np.loadtxt("no_period_found_20230409.txt").tolist()
    df = pd.read_csv(
            "/Users/u_tsubasa/Downloads/exofop_tess_tois_full_20220913.csv"
        )
    df = df[~(df["Stellar Mass (M_Sun)"] == "")]
    df = df[~(df["Stellar Radius (R_Sun)"] == "")]
    df = df[~(df["Planet Radius (R_Earth)"] == "")]
    df = df[~(df["Duration (hours)"] == "")]
    df = df.sort_values("Planet SNR", ascending=False)
    df = df.set_index(["TOI"])
    #df = df.drop(index=done_tois, errors="ignore")
    #df = df.drop(index=nodata_tois, errors="ignore")
    #df = df.drop(index=noperiod_tois, errors="ignore")
    df = df.reset_index()
    df["TOI"] = df["TOI"].astype(str)
    TOIlist = df["TOI"]
    TOIlist = done_tois
    txt_df = pd.read_table("./TOI_a_b_depth_sigma.txt", sep=" ", names=["TOI", "a_m", "b", "depth", "sigma"])
    modify_a_b_depth_sigma_txt(529.01, df, txt_df)
    for TOI in TOIlist:
        txt_df = modify_a_b_depth_sigma_txt(TOI, df, txt_df)
    txt_df.to_csv("./new_TOI_a_b_depth_sigma.txt", sep=" ", header=False, index=False)
    # txt_dfをbの列でソートする
    txt_df = txt_df.sort_values(by=["b"], ascending=False)

    src_datas = list(map(lambda x: [x, df], TOIlist))

    with Pool(cpu_count() - 6) as p:
        p.map(extract_a_b_depth_sigma_txt_wrapper, src_datas)


    
    """
