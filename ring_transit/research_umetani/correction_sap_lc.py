import os
import pdb
from concurrent import futures
from glob import glob
from multiprocessing import Pool, cpu_count

import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.table import vstack

import ring_planet


def correct_sap_lc(TOI, df, HOMEDIR):
    print(TOI)
    param_df = df[df["TOI"] == float(TOI[3:])]
    period = param_df["Period (days)"].values[0]
    mid_transit_time = (
        param_df["Transit Epoch (BJD)"].values[0] - 2457000.0
    )  # translate BTJD
    # get search result
    search_result = lk.search_lightcurve(
        f"TOI {TOI[3:-3]}",
        mission="TESS",
        cadence="short",
        author="SPOC",
    )
    csv_list = glob(f"{HOMEDIR}/obs_t0/{TOI}/*.csv")

    for i in range(len(search_result)):
        sector_lc = search_result[i].download().remove_nans()
        for csv in csv_list:
            each_table = ascii.read(csv)
            each_lc = lk.LightCurve(data=each_table)
            # timeが最も０に近いときのtime_original valueを取得
            if (
                each_lc.time_original[0] > sector_lc.time[0].value
                and each_lc.time_original[-1] < sector_lc.time[-1].value
            ):
                # each_lc.time_original[0]からeach_lc.time_original[-1]の時間でpdcsap_lcを切り出す
                epoch_start = each_lc.time_original[0]
                epoch_end = each_lc.time_original[-1]
                tmp = sector_lc[sector_lc.time.value >= epoch_start]
                pdcsap_lc = tmp[tmp.time.value <= epoch_end]
                # もともとの中央値をかける
                sap_flux_median = np.median(pdcsap_lc.sap_flux)
                each_lc.flux = each_lc.flux * sap_flux_median
                each_lc.flux_err = each_lc.flux_err * sap_flux_median

                # 補正のためのCROWDSAP、FLFRCSAPを定義
                CROWDSAP = pdcsap_lc.hdu[1].header["CROWDSAP"]
                FLFRCSAP = pdcsap_lc.hdu[1].header["FLFRCSAP"]

                # 補正するlightcurveの中央値
                median_flux = np.median(each_lc.flux.value)
                excess_flux = (1 - CROWDSAP) * median_flux
                flux_removed = each_lc.flux.value - excess_flux
                flux_corr = flux_removed / FLFRCSAP
                flux_err_corr = each_lc.flux_err.value / FLFRCSAP
                lc_corr = lk.LightCurve(
                    time=each_lc.time.value,
                    flux=flux_corr,
                    flux_err=flux_err_corr,
                )
                pdcsap_lc = pdcsap_lc.fold(
                    period=period, epoch_time=mid_transit_time
                )
                pdcsap_lc = pdcsap_lc.normalize()
                lc_corr = lc_corr.normalize()
                # lc_corrを別フォルダに保存
                os.makedirs(f"{HOMEDIR}/correct_flux/{TOI}", exist_ok=True)
                lc_corr.to_csv(
                    f"{HOMEDIR}/correct_flux/{TOI}/{csv.split('/')[-1]}",
                    overwrite=True,
                )
                # plotも保存
                savedir = HOMEDIR.replace("data", "figure").replace(
                    "lightcurve/", ""
                )
                ax = pdcsap_lc.errorbar(
                    label="PDC-SAP", color="r", alpha=0.5, marker="."
                )
                pdcsap_lc.flux = pdcsap_lc.sap_flux
                pdcsap_lc.flux_err = pdcsap_lc.sap_flux_err
                sap_lc = pdcsap_lc.normalize()
                sap_lc.errorbar(
                    ax=ax, label="SAP", color="k", alpha=0.5, marker="."
                )
                lc_corr.errorbar(
                    ax=ax, label="corrected", color="b", alpha=0.5, marker="."
                )

                os.makedirs(
                    f"{savedir}/correct_flux/{TOI}",
                    exist_ok=True,
                )
                plt.savefig(
                    f"{savedir}/correct_flux/{TOI}/{csv.split('/')[-1].split('.csv')[0]}.png"
                )
                plt.close()
            elif (
                each_lc.time_original[0] < sector_lc.time[0].value
                or each_lc.time_original[-1] > sector_lc.time[-1].value
            ):
                continue
    # fold lightcurve using all csv_data
    folded_lc = ring_planet.folding_lc_from_csv(
        f"{HOMEDIR}/correct_flux/{TOI}"
    )
    # outliersを除去
    outliers = []
    while True:
        fold_res = ring_planet.transit_fitting(
            folded_lc,
            np.nan,
            period,
            fitting_model=ring_planet.no_ring_transitfit,
        )
        transit_model = ring_planet.no_ring_transitfit(
            fold_res.params,
            folded_lc.time.value,
            folded_lc.flux.value,
            folded_lc.flux_err.value,
            list(fold_res.params.keys()),
            return_model=True,
        )
        outlier_bools = ring_planet.detect_outliers(folded_lc, transit_model)
        inverse_mask = np.logical_not(outlier_bools)

        if np.all(inverse_mask):
            # save folded_lc
            savedir = HOMEDIR.replace("each_lc", "folded_lc")
            os.makedirs(f"{savedir}/correct_flux/", exist_ok=True)
            folded_lc.to_csv(
                f"{savedir}/correct_flux/{TOI}.csv", overwrite=True
            )

            # plot folded_lc
            savedir = HOMEDIR.replace("data", "figure").replace(
                "each_lc", "folded_lc"
            )
            os.makedirs(f"{savedir}/correct_flux", exist_ok=True)
            ax = folded_lc.errorbar(
                label="corrected", color="b", alpha=0.5, marker="."
            )
            folded_table = ascii.read(
                f"/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/SAP_fitting_result/data/folded_lc/lightcurve/obs_t0/{TOI}.csv"
            )
            folded_obst0_lc = lk.LightCurve(data=folded_table)
            folded_obst0_lc.errorbar(
                ax=ax, color="k", label="sap", alpha=0.5, marker="."
            )
            try:
                outliers = vstack(outliers)
            except ValueError:
                pass
            else:
                outliers.errorbar(
                    ax=ax, color="r", label="outliers", marker="."
                )
            plt.savefig(f"{savedir}/correct_flux/{TOI}.png")
            plt.close()
            break
        else:
            outliers.append(folded_lc[outlier_bools])
            folded_lc = ring_planet.clip_outliers(
                folded_lc,
                outlier_bools,
            )


def correct_sap_lc_wrapper(args):
    return correct_sap_lc(*args)


def main():
    df = pd.read_csv(
        "/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/exofop_tess_tois_2022-09-13.csv"
    )
    HOMEDIR = "/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/SAP_fitting_result/data/each_lc/lightcurve/"
    # Load the data
    TOIlist = os.listdir(f"{HOMEDIR}/obs_t0")
    TOIlist.sort()
    pdb.set_trace()
    TOIlist = [TOI for TOI in TOIlist if TOI != ".DS_Store"][:18]
    src_datas = list(
        map(
            lambda x: [
                x,
                df,
                HOMEDIR,
            ],
            TOIlist,
        )
    )
    # for TOI in TOIlist:
    #    correct_sap_lc(TOI, df, HOMEDIR)
    batch_size = cpu_count() - 1  # バッチサイズの定義
    batches = [
        src_datas[i : i + batch_size]
        for i in range(0, len(src_datas), batch_size)
    ]
    with futures.ProcessPoolExecutor(max_workers=batch_size) as executor:
        for batch in batches:
            _ = executor.map(correct_sap_lc_wrapper, batch, timeout=None)


if __name__ == "__main__":
    main()
