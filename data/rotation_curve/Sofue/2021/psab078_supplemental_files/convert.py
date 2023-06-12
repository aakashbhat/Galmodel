from astropy.io import ascii
from astropy.table import Table
import matplotlib.pyplot as plt
data = Table.read("RC-innerMW-dR100pc.ascii",format='ascii')
print(data)
data2=Table()
data2["vc"]=data["VRot"]*(8.0/8.178) - (9*data["R"]/8.178)
data2["R"] =data["R"]
data2["sigmaminus"]=abs(data["Vrot-sigmaV"]-data["VRot"])
data2["sigmaplus"]=data["Vrot+sigmaV"]-data["VRot"]
ascii.write(data2, 'rotcurve_sofue_6kpc.ascii', overwrite=True)
plt.errorbar(data2["R"],data2["vc"],yerr=data2["sigmaplus"])
plt.xlabel("R (kpc)")
plt.ylabel("Vrot (km/s)")
plt.savefig("rotcurve_sofue_6kpc.pdf")
