import os

poly2 = os.listdir('/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/fitting_result/data/folded_lc/2poly/obs_t0/csv')
poly3 = os.listdir('/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/fitting_result/data/folded_lc/3poly/obs_t0/csv')
poly4 = os.listdir('/Users/u_tsubasa/work/ring_planet_research/ring_transit/research_umetani/fitting_result/data/folded_lc/4poly/obs_t0/csv')

result2_3 = list(set(poly2) - set(poly3))
result2_4 = list(set(poly2) - set(poly4))

print(result2_3)
print(result2_4)
import pdb;pdb.set_trace()

