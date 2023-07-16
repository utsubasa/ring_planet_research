import os
import pdb
from glob import glob

import lightkurve as lk
import numpy as np
import pandas as pd
from astropy.io import ascii

import ring_planet

HOMEDIR = os.getcwd()


def main():
    toilist = glob(
        HOMEDIR + "/SAP_fitting_result_20230510/folded_lc/obs_t0/data/*"
    )
    toilist.sort()
    sigma_list = []
    depth_list = []
    toi_list = []
    for toi in toilist:
        print(toi)
        load_dur_per_file = f"{HOMEDIR}/SAP_fitting_result_20230510/folded_lc/modelresult/calc_t0/{toi.split('/')[-1][:-4]}_folded.txt"
        with open(load_dur_per_file, "r") as f:
            # durationline = f.readlines()[-5:].split(' ')[-1]
            durationline, _, _, periodline, _ = f.readlines()[-5:]
            durationline = durationline.split(" ")[-1]
            duration = np.float(durationline)
        # read csv
        folded_table = ascii.read(toi)
        folded_lc = lk.LightCurve(data=folded_table)
        folded_lc = folded_lc[
            (folded_lc.time.value < duration * 0.7)
            & (folded_lc.time.value > -duration * 0.7)
        ]
        if len(folded_lc.time) < 500:
            t = folded_lc.time.value
            flux_data = folded_lc.flux.value
            flux_err_data = folded_lc.flux_err.value
        else:
            t, flux_data, flux_err_data = ring_planet.binning_lc(folded_lc)
        # get sigma
        sigma = np.mean(flux_err_data)
        # get depth
        depth = np.min(flux_data)
        sigma_list.append(sigma)
        depth_list.append(depth)
        toi_list.append(toi.split("/")[-1][:-4])
    # make dataframe
    df = pd.DataFrame(
        {
            "TOI": toi_list,
            "depth": depth_list,
            "sigma": sigma_list,
        }
    )
    # save
    df.to_csv(
        f"{HOMEDIR}/oversn100_tois_depth_sigma.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
