import sys
import numpy as np
from fit_model import model
import matplotlib.pyplot as plt
# set the values of new parameters
def make_dic(names, values):
    dic = {}
    for (i, name) in enumerate(names):
        dic[name] = values[i]
        print(name, values[i])
    return dic

def name_load(filename):
    file_now = open(filename)
    lines = file_now.readlines()
    names = []
    for line in lines:
        itemList = line.split()
        names.append(itemList[0])
    return names


# read the data file (xdata, ydata, yerr)
datafile = "./test/Q8_l_KIC10403228_test_detrend.dat"
dataarr  = np.loadtxt(datafile).T
xdata = dataarr[0,:]
ydata = dataarr[1,:]
yerr  = dataarr[2,:]

names = name_load("./test/parname.dat")
parfile  = "./test/para_result_ring.dat"
parvalues = []
file = open(parfile, "r")
lines = file.readlines()
for (i, line) in enumerate(lines):

    itemList = line.split()
    parvalues.append(float(itemList[1]))


parvalues = np.array(parvalues)


# print test data
#""" 
pdic = make_dic(names, parvalues)
ymodel = model(xdata, pdic)
plt.plot(xdata, ymodel)
plt.plot(xdata, ydata)
plt.show()



