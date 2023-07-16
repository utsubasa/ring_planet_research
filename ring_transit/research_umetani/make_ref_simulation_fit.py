import os
import pdb
import sys
from concurrent import futures

import numpy as np
import pandas as pd
from tqdm import tqdm

import ring_planet


def main():
    oridf = pd.read_csv("./exofop_tess_tois_20230526_sigma.csv")
    df = oridf
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
    theta = 45
    phi = 45
    min_flux = 0.99
    bin_error_list = np.arange(0.0001, 0.0041, 0.0001)
    bin_error_list = np.around(bin_error_list, decimals=4)
    bin_error = 0.0001
    r_out_list = np.arange(1.1, 2.1, 0.1)
    r_out_list = np.around(r_out_list, decimals=1)
    # new_bin_error_list = []

    for b in tqdm(b_list):
        # 1.1から5.0までの範囲で、0.1刻みでループ.表記は小数点第一位まで
        src_datas = list(
            map(
                lambda x: [
                    bin_error,
                    b,
                    theta,
                    phi,
                    period,
                    min_flux,
                    x,
                ],
                r_out_list,
            )
        )
        batch_size = 4  # バッチサイズの定義
        batches = [
            src_datas[i : i + batch_size]
            for i in range(0, len(src_datas), batch_size)
        ]
        with futures.ProcessPoolExecutor(max_workers=4) as executor:
            for batch in batches:
                _ = executor.map(
                    ring_planet.process_bin_error_wrapper,
                    batch,
                    timeout=None,
                )
            """
            if os.path.isfile(
                f"./target_selection/figure/r_out_{r_out}/b_{b}/{theta}deg_{phi}deg/{min_flux}_{bin_error}.png"
            ):
                print(
                    f"r_out_{r_out}/b_{b}/{theta}deg_{phi}deg/{min_flux}_{bin_error}.png is exist."
                )
                continue
            else:
                new_bin_error_list.append(bin_error)

            if len(new_bin_error_list) == 0:
                continue
            """
            """
            for bin_error in new_bin_error_list:
                process_bin_error(
                    bin_error, b, rp_rs, theta, phi, period, min_flux, r_out
                )
            
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
                        r_out,
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
                    _ = executor.map(
                        ring_planet.process_bin_error_wrapper,
                        batch,
                        timeout=None,
                    )
            """
    sys.exit()
    """
    #for theta in theta_list:
    #    for phi in phi_list:
            theta = 45
            phi = 45
            min_flux = 0.99
            bin_error_list = np.arange(0.0001, 0.0041, 0.0001)
            bin_error_list = np.around(bin_error_list, decimals=4)

            new_bin_error_list = []

            for bin_error in bin_error_list:
                if os.path.isfile(
                    f"./target_selection/r_out_{r_out}/figure/b_{b}/{theta}deg_{phi}deg/{min_flux}_{bin_error}.png"
                ):
                    print(
                        f"r_out_{r_out}/b_{b}/{theta}deg_{phi}deg/{min_flux}_{bin_error}.png is exist."
                    )
                    continue
                else:
                    new_bin_error_list.append(bin_error)

            if len(new_bin_error_list) == 0:
                continue

            # get rp_rs value from transit depth
            rp_rs = get_rp_rs(min_flux, b, period, theta, phi)

            for bin_error in new_bin_error_list:
                process_bin_error(
                    bin_error, b, rp_rs, theta, phi, period, min_flux, r_out
                )

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
                        r_out
                    ],
                    new_bin_error_list,
                )
            )
            print(b, theta, phi, min_flux, bin_error)
            batch_size = 3  # バッチサイズの定義
            batches = [src_datas[i:i+batch_size] for i in range(0, len(src_datas), batch_size)]
            with futures.ProcessPoolExecutor(max_workers=3) as executor:
                for batch in batches:
                    future_list = executor.map(
                        process_bin_error_wrapper, batch, timeout=None
                    )

            sys.exit()
    """


if __name__ == "__main__":
    main()
