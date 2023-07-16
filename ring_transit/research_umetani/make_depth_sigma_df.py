import concurrent.futures
import os
import pdb

import lightkurve as lk
import numpy as np
import pandas as pd
from astropy.io import ascii
from tqdm import tqdm

import ring_planet

HOMEDIR = os.getcwd()


def get_sigma(toi: float, df: pd.DataFrame):
    target_row = df.loc[df["TOI"] == toi]
    period = target_row["Period (days)"].values[0]
    duration = target_row["Duration (hours)"].values[0]

    # calculate sigma using cdpp
    sigma = ring_planet.calc_sigma(
        period, duration, toi, restrict_sector=False
    )

    return sigma


def main():
    """Make dataframe have the column of depth and sigma. use csv file"""
    # read csv file
    df = pd.read_csv(f"{HOMEDIR}/exofop_tess_tois_20230526_sigma.csv")
    # get depth
    # depth_list = 1 - (df["Depth (ppm)"] * 1e-6)
    # pdb.set_trace()
    # df["depth"] == depth_list

    toilist = df["TOI"].values
    # toilistからno_data_found_20230526.txtに含まれるtoiを除外
    with open("no_data_found_20230526.txt", "r") as f:
        no_data_found = f.readlines()
    no_data_found = [float(toi.strip("\n")) for toi in no_data_found]
    toilist = [toi for toi in toilist if toi not in no_data_found]
    # calculate sigma
    col_name = "sigma_full"
    for toi in tqdm(toilist):
        if df[df["TOI"] == toi][col_name].values[0] != 0:
            print(f"already calculated: {toi}")
            continue
        sigma = get_sigma(toi, df)
        if sigma is None:
            continue
        df.at[df["TOI"] == toi, col_name] = sigma
        df.to_csv("exofop_tess_tois_20230526_sigma.csv", index=False)
        print(f"TOI: {toi}, sigma: {sigma}")

    pdb.set_trace()
    # read


if __name__ == "__main__":
    # ring_planet.calc_sigma(period=4.41, duration=3.416, toi="114.01")
    main()
