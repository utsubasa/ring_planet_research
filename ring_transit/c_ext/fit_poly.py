import pylab as pl
import numpy as np
import scipy.optimize as optimization
import sys
import csv

def func_array(x, *a):
    sum = 0
    
    #for j in xrange(len(x)):
    for (i, now) in enumerate(a):
        if(i>2):
            break
        sum += now * (x** i)
    return sum

datarr = np.loadtxt(sys.argv[1]).T
x = np.array(datarr[0,:])
y = np.array(datarr[3,:] )
yerr = np.array(datarr[4,:])
x0 = np.zeros(5)
#print x

transit_haba =1.5
data_haba = 3.09
transit_center = 744.83-0.057
mean =0
count = 0
x_red = []
y_red = []
yerr_red = []

x_fit =[]
y_fit =[]
yerr_fit = []
for a in y:
    print a

for (i,a) in enumerate(x):
    ## put some restrictions on the t_range
    #if ( y[i] == "nan" or x[i] == "nan"):
    if(np.isnan(y[i]) == True or np.isnan(x[i]) == True ):
        continue
    if( (x[i]> transit_center - data_haba ) and ( x[i] < transit_center + data_haba ) ):
        x_red.append(x[i])
        y_red.append(y[i])
        yerr_red.append(yerr[i])
        print x[i], y[i]
        if( (x[i] > transit_center - transit_haba ) and ( x[i] < transit_center + transit_haba)):
            continue
        mean += y[i]
        count+=1
        x_fit.append(x[i])
        y_fit.append(y[i])
        yerr_fit.append(yerr[i])


mean/=count

x_fit = np.array(x_fit)
y_fit = np.array(y_fit)
yerr_fit = np.array(yerr_fit)

for (i,a) in enumerate(y_fit):

    print x_fit[i], y_fit[i], yerr_fit[i]
    x_fit[i] = x_fit[i] - transit_center
    y_fit[i] = y_fit[i]/mean
    yerr_fit[i] = yerr_fit[i]/mean

#z = np.polyfit(x,y,2, cov = yerr)
popt, pcov =  optimization.curve_fit(func_array, x_fit, y_fit, x0, yerr_fit,absolute_sigma = True)
perr = np.sqrt(np.diag(pcov))
print pcov
#print perr

flux_norm = popt[0]

for (i,a) in enumerate(popt):
    popt[i] /= flux_norm
    hamu =0
    if( i>2):
        hamu = 1
    print "%d %f %d %f %f" %(i+14,  popt[i], hamu,  popt[i]-0.01, popt[i] + 0.01)





file_name = sys.argv[1].split(".")[0] + "_test_detrend.dat"

file_write = open(file_name, "w")


val =0
for num in range(0, len(x_red)):
    x_now = x_red[num] - transit_center
    scale = mean * (popt[0] + popt[1] * x_now + popt[2] * x_now **2 )
    #scale = mean * (popt[0] + popt[1] * x_now + popt[2] * x_now **2 + popt[3] * x_now ** 3 + popt[4] * x_now **4)
    file_write.write("%f %f %f %f %f \n" % (x_red[num]-transit_center, y_red[num]/(flux_norm*mean), yerr_red[num]/(flux_norm*mean), y_red[num]/scale, yerr_red[num]/scale ) )
file_write.close()


\