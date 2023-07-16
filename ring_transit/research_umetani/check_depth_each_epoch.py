import glob
import os
import pdb
import re
import statistics

import batman
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import ring_planet


def calculate_standard_deviation(lst):
    std_dev = statistics.stdev(lst)
    return std_dev


def sort_filenames_by_number(filenames):
    def extract_number(filename):
        # ファイル名から数値の部分を抽出します
        matches = re.findall(r"\d+", filename)
        return int(matches[0]) if matches else None

    sorted_filenames = sorted(filenames, key=extract_number)
    return sorted_filenames


def main():
    time = np.linspace(-0.2, 0.2, 1000)

    toilist = glob.glob(f"{loaddir}/*")
    for toidir in tqdm(toilist):
        toi = toidir.split("/")[-1]
        param_df = df[df["TOI"] == float(toi[3:])]
        depth_col = param_df["Depth (ppm)"].values[0]
        ref_depth = 1 - (depth_col / 1e6)
        txtlist = glob.glob(f"{toidir}/*")
        if len(txtlist) == 0:
            continue
        depth_list = []
        depth_err_list = []
        for txt in txtlist:
            with open(txt, "r") as f:
                lines = f.readlines()
                # linesから12-15行目、18,19行目を抜き出す
                per = float(lines[11].split("    per:  ")[1].split(" ")[0])
                rp = float(lines[12].split("    rp:   ")[1].split(" ")[0])
                a = float(lines[13].split("    a:    ")[1].split(" ")[0])
                b = float(lines[14].split("    b:    ")[1].split(" ")[0])
                q1 = float(lines[17].split("    q1:   ")[1].split(" ")[0])
                q2 = float(lines[18].split("    q2:   ")[1].split(" ")[0])
            # depthを計算する
            params = ring_planet.noring_params_setting(
                per, rp, t0=0, a_rs=a, b=b, q1=q1, q2=q2
            )
            params_batman = ring_planet.set_params_batman(
                params, list(params.valuesdict().keys())
            )
            m = batman.TransitModel(params_batman, time)  # initializes model
            model = m.light_curve(params_batman)
            depth = np.min(model)
            depth_list.append(depth)
        # depth_listの各要素の標準偏差×3をy軸の範囲にする
        ave_depth = statistics.mean(depth_list)
        std_dev = calculate_standard_deviation(depth_list)
        """
        data_length = len(depth_list)
        chisq = np.sum(
            ((np.array(depth_list) - ave_depth) / np.array(depth_err_list))
            ** 2
        )
        redchi = chisq / (data_length - 1)
        print(f"{toi}:  {redchi:2f} = {int(chisq)} / {data_length}")
        """

        # plot depth and depth_err
        plt.scatter(
            range(len(depth_list)),
            depth_list,
            color="k",
        )
        # 標準偏差で2シグマ、3シグマごとに線を引く
        plt.hlines(
            ave_depth + std_dev * 2,
            0,
            len(depth_list),
            "orange",
            linestyles="dashed",
            label="2 sigma",
            alpha=0.5,
        )
        plt.hlines(
            ave_depth - std_dev * 2,
            0,
            len(depth_list),
            "orange",
            linestyles="dashed",
            alpha=0.5,
        )
        plt.hlines(
            ave_depth + std_dev * 3,
            0,
            len(depth_list),
            "r",
            linestyles="dashed",
            label="3 sigma",
            alpha=0.5,
        )
        plt.hlines(
            ave_depth - std_dev * 3,
            0,
            len(depth_list),
            "r",
            linestyles="dashed",
            alpha=0.5,
        )

        plt.legend()
        plt.title(
            f"{toi} mean D: {ave_depth:.4f}, pdc D: {ref_depth:.4f} std D: {std_dev:.4f}"
        )
        plt.xlabel("epoch number")
        plt.ylabel("transit depth")
        if ave_depth - std_dev * 4 < 0:
            plt.ylim(0, ave_depth + std_dev * 4)
        else:
            plt.ylim(ave_depth - std_dev * 4, ave_depth + std_dev * 4)
        plt.savefig(f"{savedir}/{toi}.png")
        plt.close()


if __name__ == "__main__":
    homedir = os.getcwd() + "/SAP_fitting_result_tpfcorrect_20230510"
    type = "tpf"
    # loaddir = f"{homedir}/SAP_fitting_result_{type}correct/each_lc/modelresult/calc_t0"
    # savedir = f"{homedir}/SAP_fitting_result_{type}correct/check_depth"
    loaddir = f"{homedir}/each_lc/modelresult/calc_t0"
    savedir = f"{homedir}/check_depth"
    os.makedirs(savedir, exist_ok=True)
    df = pd.read_csv(os.getcwd() + "/exofop_tess_tois_2022-09-13.csv")
    main()
