import os

import pandas as pd

data_list = os.listdir(
    f"/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/fitting_result/figure/each_lc/2ndloop/"
)
os.chdir(
    "/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/fitting_result/figure/each_lc/2ndloop/"
)
epoch_num_list = []
for toi in data_list:
    epoch_list = os.listdir(toi)
    epoch_num_list.append(len(epoch_list))
pd.DataFrame({"TOI": data_list, "epoch_num": epoch_num_list}).to_csv(
    f"/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/epoch_num.csv",
    index=False,
)
