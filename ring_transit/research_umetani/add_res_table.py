import os
import pdb
from decimal import Decimal

import pandas as pd

# fit_reportディレクトリを読み込む
path = os.getcwd()
path = path + "/fit_report"

results = os.listdir(path)
# below_p_3sigma_TOIs_b_0.9.csvから、TOIの列をリストとして取得
df = pd.read_csv("below_p_3sigma_TOIs_b_0.9.csv")
TOIs = df["TOI"].values.tolist()
df.index = df["TOI"]
# TOIsをstr型に変換
TOIs = [str(TOI) for TOI in TOIs]
for toi in TOIs:
    # txtフif ァイル(result)を読み込む
    try:
        with open(path + "/TOI" + toi + ".txt") as f:
            lines = f.readlines()
            extract_words = {
                "period": "per:  ",
                "n_bins": "# data points      = ",
                "chisq_ringless": "chi-square         = ",
                "chisq_ring": "chi-square         = ",
                "p_value": "p_value: ",
            }
            for val, string in extract_words.items():
                if val == "chisq_ring":
                    # 逆順で取得
                    line = [line for line in reversed(lines) if string in line]
                else:
                    # 抽出する文字列が含まれる行を取得
                    line = [line for line in lines if string in line]
                # 抽出する文字列が含まれる行から、抽出する文字列の後ろの値を取得
                value = line[0].split(string)[1].split(" ")[0]
                if val == "n_bins":
                    value = str(int(value))
                else:
                    # valueの小数第二位までの小数点以下の値を取得し、float型に変換
                    decimal_number = Decimal(value)
                    value = decimal_number.quantize(Decimal("0.00"))
                    value = "{:.2f}".format(value)
                # 抽出した値を辞書に格納
                extract_words[val] = value
    except FileNotFoundError:
        continue
    # extract_wordsから、"n_bins", "chisq_ringless"､ "chisq_ring"の値を取得し、（）で囲む
    fit_chisq = f"({extract_words['chisq_ringless']}, {extract_words['chisq_ring']}, {extract_words['n_bins']})"
    # それぞれの値をdf.atで代入する
    df.at[float(toi), "fit_chisq"] = fit_chisq
    df.at[float(toi), "p_value"] = extract_words["p_value"]
    df.at[float(toi), "period"] = extract_words["period"]

# dfからカラムが"TOI", "depth", "sigma", "period", "fit_chisq", "p_value"の行を抽出
df = df[["TOI", "depth", "sigma", "period", "fit_chisq", "p_value"]]
df.to_csv("try.csv", index=False)
