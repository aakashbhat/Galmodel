from astropy import coordinates as coord
from galpy import orbit
from astropy import units
from galpy.potential import MWPotential2014 as pott
import numpy as np
import scipy as sc
from lmfit import Model
import matplotlib.pyplot as plt
import pandas as pd
from astroquery.simbad import Simbad
from astropy.io import ascii
from lmfit import minimize, Parameters,fit_report
from astropy.table import vstack
import spectres


data1 = ascii.read("rotcurv_eilers.ascii")
data2 = ascii.read("rotcurv_bhatta_25kpc.ascii")
data3 = ascii.read("rotcurv_reid_1to5.ascii")
data=vstack([data1,data2,data3])

radius=data2['R']
vel=data2['vc']
poser=data3['evc']
#neger=data3['sigmaminus']
len=len(radius)

newradius=np.linspace(25.0,200.0,18)
#radiusbin=np.digitize(radius,new radius)
bin_means, bin_edges, binnumber=sc.stats.binned_statistic(radius,vel,bins=newradius)
bin_width = (bin_edges[1] - bin_edges[0])
bin_centers = bin_edges[1:] - bin_width/2
plt.errorbar(bin_centers,bin_means,fmt='+')

plt.savefig('newbinnedcurve_bhatta.pdf')



