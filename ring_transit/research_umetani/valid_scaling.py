import glob
import os
import pdb

import matplotlib.pyplot as plt
import numpy as np


def get_chisq(file):
    with open(
        file,
        mode="r",
    ) as f:
        chisquare = f.readlines()[7]

        chisquare = chisquare.split("    chi-square         = ")[-1]

    return float(chisquare)


def calc_scaling(ref_chisquare, depth, sigma, REF_DEPTH, REF_SIGMA):
    depth_scaling = ((1 - depth) / (1 - REF_DEPTH)) ** 2.5
    sigma_scaling = (sigma / REF_SIGMA) ** (-2)
    scaled_chisq = ref_chisquare * depth_scaling * sigma_scaling

    return scaled_chisq


def normalize_difference(
    depth: float, sigma: float, ref_chisquare: float, data_chisquare: float
) -> float:

    scaled_chisq = calc_scaling(
        ref_chisquare, depth, sigma, REF_DEPTH, REF_SIGMA
    )
    diff = scaled_chisq - data_chisquare
    normalized_diff = diff / scaled_chisq

    return normalized_diff


def main():
    # load depth=0.97,sigma=0.0001 data from /Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/depth_error
    b_list = os.listdir(f"{HOMEDIR}/depth_error/data/")
    b_list = [data for data in b_list if data != ".DS_Store"]
    b_list.sort()
    for b in b_list:
        deg_list = os.listdir(f"{HOMEDIR}/depth_error/data/{b}/")
        deg_list = [data for data in deg_list if data != ".DS_Store"]
        deg_list.sort()
        for deg in deg_list:
            files = glob.glob(f"{HOMEDIR}/depth_error/data/{b}/{deg}/TOI*.txt")
            # load REF_TXT_NAME
            ref_chisquare = get_chisq(
                f"{HOMEDIR}/depth_error/data/{b}/{deg}/{REF_TXT_NAME}"
            )

            # remove f"{HOMEDIR}/depth_error/data/{b}/{deg}/{REF_TXT_NAME}" from files
            files = [
                file
                for file in files
                if file
                != f"{HOMEDIR}/depth_error/data/{b}/{deg}/{REF_TXT_NAME}"
            ]
            files.sort()
            depth_list = []
            sigma_list = []
            diff_list = []
            for file in files:
                chisquare = get_chisq(file)
                depth = float(file.split("/")[-1].split("_")[1])
                sigma = float(
                    file.split("/")[-1].split("_")[2].split(".txt")[0]
                )
                depth_list.append(depth)
                sigma_list.append(sigma)
                normalized_diff = normalize_difference(
                    depth, sigma, ref_chisquare, chisquare
                )
                diff_list.append(normalized_diff)
            # depth_listの順番にdiff_listを並び替える
            depth_list, diff_list = zip(*sorted(zip(depth_list, diff_list)))

            plt.plot(depth_list, diff_list, color="black")
            # plt.colorbar(label="Difference")
            plt.xlabel("Depth")
            plt.ylabel("Difference")
            plt.title(f"{b}, {deg}")
            plt.show()
            """            # compare chisquare with ref_chisquare
            status = file.split(
                "/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/depth_error/data/"
            )[-1]
            # print(
            #    f"{status}:\ncalc_chisq: {scaled_chisq}, data_chisq: {chisquare}"
            # )
            """

            pdb.set_trace()


HOMEDIR = os.getcwd()
REF_TXT_NAME = "TOI495.01_0.99_0.0001.txt"
REF_DEPTH = float(REF_TXT_NAME.split("_")[1])
REF_SIGMA = float(REF_TXT_NAME.split("_")[2].split(".txt")[0])

if __name__ == "__main__":
    main()
