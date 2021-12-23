# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

#if __name__ ==  '__main__':
homedir = '/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani'

df = pd.read_csv(f'{homedir}/exofop_tess_tois.csv')
#df = df[df['Source'].str.contains('spoc')]
df = df[df['Planet SNR']>100]
import pdb; pdb.set_trace()
TOIlist = df['TOI']
df['Aizawa+2018SN'] = 0.0
for TOI in TOIlist:
    print(TOI)
    #tpf = lk.search_targetpixelfile('TIC {}'.format(TIC), mission='TESS', cadence="short").download()
    """
    while True:
        try:
            search_result = lk.search_lightcurve(f'TOI-{TOI}', mission='TESS', cadence="short", author='SPOC')
            #tpf_file = lk.search_targetpixelfile(f'TIC {TIC}', mission='TESS', cadence="short", author='SPOC').download_all(quality_bitmask='default')
            #tpf_file.plot()
            #plt.show()

        except HTTPError:
            print('HTTPError, retry.')

        else:
            break
    lc_collection = search_result.download_all()
    """
    try:
        period = df.at[df[df['TOI']==TOI].index[0], 'Period (days)']
    except KeyError:
        print('keyerror')
        import pdb; pdb.set_trace()
    try:
        transit_N = len(df.at[df[df['TOI']==TOI].index[0], 'Sectors'].split(','))*27/period
    except TypeError:
        print(f'No data found for target TOI {TOI}')
        continue
    df.at[df[df['TOI']==TOI].index[0], 'Aizawa+2018SN'] = np.sqrt(transit_N) * df.at[df[df['TOI']==TOI].index[0],'Depth (ppm)']/ df.at[df[df['TOI']==TOI].index[0],'Depth (ppm) error']
    continue
df.to_csv(f'{homedir}/exofop_tess_tois2.csv')
import pdb; pdb.set_trace()

'''
duration = item['Planet Transit Duration Value [hours]'] / 24
transit_time = item['Planet Transit Midpoint Value [BJD]'] - 2457000.0 #translate BTJD
transit_start = transit_time - (duration/2)
transit_end = transit_time + (duration/2)
transit_depth_val = item['Planet Transit Depth Value [ppm]']
transit_depth_upper_unc = item['Planet Transit Depth Upper Unc [ppm]']
transit_depth_lower_unc = item['Planet Transit Depth Lower Unc [ppm]']
'''
