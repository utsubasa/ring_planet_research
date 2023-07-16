import pdb

import numpy as np
import pandas as pd


def calc_mp_ms(rp, rs):
    rp_rs = rp / rs
    mp_ms = 0.337 * rp_rs ** (1 / 0.53) * F

    return mp_ms


def calc_upper_p_orb(mp_ms, a, rp):
    A = (3 * kp) / (2 * c * Qp)
    B = (a / rp) ** -3
    C = 1 / mp_ms
    p_orb = (13 * (10**9) * A) * B * C * 2 * np.pi

    return p_orb


c = 0.23
Qp = 10**6.5
kp = 1.5
F = 5.46 * 10**8
j2 = 0.016298


# koiのテーブルを読み込む
koi = pd.read_csv(
    "/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/koi_cumulative_2023.07.14_02.15.52.csv",
    index_col=0,
)

# koi_disposition==CONFIRMEDのものを抽出
koi_confirmed = koi[koi["koi_disposition"] == "CONFIRMED"]

# koi_pradとkoi_sradを使い、calc_mp_s(rp, rs)の計算結果をkoi_confirmedに追加
koi_confirmed["mp_ms"] = calc_mp_ms(
    koi_confirmed["koi_prad"], koi_confirmed["koi_srad"]
)

# mp_ms, koi_pradとkoi_smaを使って、calc_upper_p_orb(mp_ms, a, rp)の計算結果をkoi_confirmedに追加
koi_confirmed["limit_p_orb"] = calc_upper_p_orb(
    koi_confirmed["mp_ms"], koi_confirmed["koi_sma"], koi_confirmed["koi_prad"]
)

# koi_periodとlimit_p_orbを比較し、koi_periodがlimit_p_orbよりも大きいものを抽出
koi_confirmed["koi_period"] = koi_confirmed["koi_period"].astype(float)
koi_confirmed["limit_p_orb"] = koi_confirmed["limit_p_orb"].astype(float)

koi_confirmed["is_ok"] = (
    koi_confirmed["koi_period"] > koi_confirmed["limit_p_orb"]
)

# is_okがTrueのものを抽出
koi_confirmed_ok = koi_confirmed[koi_confirmed["is_ok"] == True]

# koi_periodが10以下のものを抽出
koi_confirmed_ok_10 = koi_confirmed_ok[koi_confirmed_ok["koi_period"] <= 10]
