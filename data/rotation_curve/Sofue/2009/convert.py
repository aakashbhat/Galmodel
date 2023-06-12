from astropy.io import ascii
from astropy.table import Table
import matplotlib.pyplot as plt
data = Table.read("RC-innerMW.ascii",format='ascii')
print(data)
data2=Table()
data2["vc"]=data["VRot"]*(8.0/8.178) + (29/8.178)*data["R"]
data2["R"] =data["R"]
data2["sigmaminus"]=abs(data["Vrot-sigmaV"]-data["VRot"])
data2["sigmaplus"]=data["Vrot+sigmaV"]-data["VRot"]
ascii.write(data2, 'rotcurv_sofue_2009_5kpc.ascii', overwrite=True)
plt.errorbar(data2["R"],data2["vc"],fmt='o',yerr=data2["sigmaplus"])
plt.xlabel("R (kpc)")
plt.ylabel("Vrot (km/s)")
plt.savefig("rotcurv_sofue_2009_5kpc.pdf")
