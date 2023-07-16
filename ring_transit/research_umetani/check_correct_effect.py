import os

import lightkurve as lk
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import ascii

HOMEDIR = os.getcwd()
TOI = "TOI107.01"


def check_correct_effect():
    folded_table = ascii.read(
        f"/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/archive_SAP_fitting_result/SAP_fitting_result/data/folded_lc/lightcurve/obs_t0/{TOI}.csv"
    )
    original_fold_lc = lk.LightCurve(data=folded_table)
    folded_table = ascii.read(
        f"/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/SAP_fitting_result_20230510/folded_lc/obs_t0/data/{TOI}.csv"
    )
    new_fold_lc = lk.LightCurve(data=folded_table)

    # plot original_fold_lc and new_fold_lc
    ax = original_fold_lc.errorbar(
        label="original", color="b", alpha=0.5, marker=".", zorder=2
    )
    new_fold_lc.errorbar(
        ax=ax, label="corrected", color="r", alpha=0.5, marker=".", zorder=1
    )
    ax.legend()
    plt.title(TOI)
    plt.show()


def compare_variance():
    csv_list = os.listdir(
        f"{HOMEDIR}/SAP_fitting_result_under_p0.01_20230523/folded_lc/obs_t0/data"
    )
    df = pd.read_csv("exofop_tess_tois_20230526_sigma.csv")
    for csv in csv_list:
        folded_table = ascii.read(
            f"{HOMEDIR}/SAP_fitting_result_under_p0.01_20230523/folded_lc/obs_t0/data/{csv}"
        )
        fold_lc = lk.LightCurve(data=folded_table)
        # read table csv
        toi = csv.split(".csv")[0].split("TOI")[1]
        try:
            cdpp_sigma = df[df["TOI"] == float(toi)][
                "sigma_sap_corrected"
            ].values[0]
        except:
            continue
        print(f"TOI: {toi}")
        print(f"anal sigma: {fold_lc.flux.std()}")
        print(f"cdpp sigma: {cdpp_sigma}\n")


if __name__ == "__main__":
    compare_variance()
