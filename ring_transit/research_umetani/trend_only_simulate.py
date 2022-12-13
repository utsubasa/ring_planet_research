# -*- coding: utf-8 -*-
# import astropy.units as u
import os
import pdb
import time
import warnings

import batman
import lightkurve as lk
import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.table import vstack
from scipy.stats import linregress, t
import c_compile_ring

warnings.filterwarnings("ignore")

def make_simulation_data():
    """make_simulation_data"""
    t = np.linspace(-0.3, 0.3, 500)
    #t = t + np.random.randn()*0.01
    ymodel = 1+ np.random.randn(len(t))*0.001 + np.sin( (t/0.6 +1.2*np.random.rand()) * np.pi)*0.01
    yerr = np.array(t/t)*1e-3
    each_lc = lk.LightCurve(t, ymodel, yerr)
    
    return each_lc

def curve_fitting(each_lc, res=None):
    out_transit = each_lc[
        (each_lc["time"].value < -0.07)
        | (each_lc["time"].value > 0.07)
    ]
    model = lmfit.models.PolynomialModel(degree=4)
    # poly_params = model.make_params(c0=0, c1=0, c2=0, c3=0, c4=0, c5=0, c6=0, c7=0)
    poly_params = model.make_params(c0=1, c1=0, c2=0, c3=0, c4=0)
    result = model.fit(
        out_transit.flux.value, poly_params, x=out_transit.time.value
    )
    result.plot()
    os.makedirs(f"{SAVE_DIR}/curvefit", exist_ok=True)
    plt.savefig(f"{SAVE_DIR}/curvefit/{i}.png")
    plt.close()

    return result

def curvefit_normalize(each_lc, poly_params):
    poly_model = np.polynomial.Polynomial(
        [
            poly_params["c0"].value,
            poly_params["c1"].value,
            poly_params["c2"].value,
            poly_params["c3"].value,
            poly_params["c4"].value,
        ]
    )
    # normalization
    each_lc.flux = each_lc.flux.value / poly_model(each_lc.time.value)
    each_lc.flux_err = each_lc.flux_err.value / poly_model(each_lc.time.value)
    
    return each_lc

EPOCH_NUM = 54
SAVE_DIR = f'/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/fitting_result/data/only_trend'
num_list = np.arange(EPOCH_NUM)



for i in range(EPOCH_NUM):
    print(f"preprocessing...epoch: {i}")
    each_lc = make_simulation_data()
    curvefit_res = curve_fitting(each_lc)
    t = each_lc.time.value
    poly_params = curvefit_res.params.valuesdict()
    poly_model = np.polynomial.Polynomial(
        [poly_params["c0"], poly_params["c1"], poly_params["c2"], poly_params["c3"], poly_params["c4"]]
    )
    polynomial_model = poly_model(t)
    """
    fig = plt.figure()
    ax_lc = fig.add_subplot(2,1,1) #for plotting transit model and data
    ax_re = fig.add_subplot(2,1,2) #for plotting residuals
    ax_lc.plot(
                    t,
                    polynomial_model,
                    label="polynomial model",
                    ls="-.",
                    color="red",
                    alpha=0.5,
                )
    each_lc.errorbar(ax=ax_lc)
    residuals = each_lc - polynomial_model

    residuals.plot(ax=ax_re, color='blue', alpha=0.3,  marker='.', zorder=1)
    ax_re.plot(t, np.zeros(len(t)), color='black', zorder=2)
    ax_lc.legend()
    ax_lc.set_title(f'chisq:{np.sum(((each_lc.flux.value - polynomial_model)/each_lc.flux_err.value)**2):.0f}/{500:.0f}')
    plt.tight_layout()
    plt.show()
    import pdb;pdb.set_trace()
    """
    after_curvefit_lc = curvefit_normalize(each_lc, curvefit_res.params)
    os.makedirs(f'{SAVE_DIR}/each_lc', exist_ok=True)
    after_curvefit_lc.write(f'{SAVE_DIR}/each_lc/{i}.csv')

each_lc_list = []
total_lc_csv = os.listdir(f"{SAVE_DIR}/each_lc/")
for each_lc_csv in total_lc_csv:
    each_table = ascii.read(f"{SAVE_DIR}/each_lc/{each_lc_csv}")
    each_lc = lk.LightCurve(data=each_table)
    each_lc_list.append(each_lc)
cleaned_lc = vstack(each_lc_list)
cleaned_lc.sort("time")
cleaned_lc = cleaned_lc.bin(bins=500).remove_nans()
fig = plt.figure()
ax_lc = fig.add_subplot(2,1,1) #for plotting transit model and data
ax_re = fig.add_subplot(2,1,2) #for plotting residuals
ax_lc.plot(
                cleaned_lc.time.value,
                np.ones(len(cleaned_lc)),
                label="y=1",
                ls="-.",
                color="red",
                alpha=0.5,
            )
cleaned_lc.errorbar(ax=ax_lc)
residuals = cleaned_lc - 1

residuals.plot(ax=ax_re, color='blue', alpha=0.3,  marker='.', zorder=1)
ax_re.plot(cleaned_lc.time.value, np.zeros(len(cleaned_lc.time.value)), color='black', zorder=2)
ax_lc.legend()
ax_lc.set_title(f'chisq:{np.sum(((cleaned_lc.flux.value - 1)/cleaned_lc.flux_err.value)**2):.0f}/{500:.0f}')
plt.tight_layout()
plt.show()

print(np.sum(((cleaned_lc.flux.value - 1)/cleaned_lc.flux_err.value)**2))