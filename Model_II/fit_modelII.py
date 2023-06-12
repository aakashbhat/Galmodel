#from astropy import astropy.units as u
from astropy import coordinates as coord
from galpy import orbit
from astropy import units
from galpy.potential import MWPotential2014 as pott
import numpy as np

from lmfit import Model
import matplotlib.pyplot as plt
import pandas as pd
from astroquery.simbad import Simbad
from astropy.io import ascii
from lmfit import minimize, Parameters,fit_report
from astropy.table import vstack
###################
#Define Rotation Curves for 3 components:
##################


def bulgerot(R,Mb,bb):
    vr2=Mb*R*R/pow((R*R+bb*bb),1.5)
    return vr2
def diskrot(R,z,Md,ad,bd):
    zt=pow(z*z+bd*bd,0.5)+ad
    zt=zt*zt
    den=pow((R*R+zt),1.5)
    vr2=Md*R*R/den
    return vr2
def halorot(R,Mh,ah):
    #vr2=np.empty(len(R))
    vr2=Mh/pow((R*R+ah*ah),0.5)
    return vr2



#########################
#Read Data from files:
########################


data1 = ascii.read("rotcurv_eilers.ascii")
data2 = ascii.read("rotcurv_bhatta_25kpc.ascii")
data=vstack([data1,data2])

radius1=data1['R']
vel1=data1['vc']
poser1=data1['sigmaplus']
neger1=data1['sigmaminus']

radius2=data2['R']
vel2=data2['vc']
poser2=data2['sigmaplus']
neger2=data2['sigmaminus']


radius=data['R']
vel=data['vc']
poser=data['sigmaplus']
neger=data['sigmaminus']

########################
#fitmodel:
#######################
#adiusformodel=np.linspace(0,50.0,1000)
def rotcurve(R,mb,bb,md,ad,bd,mh,ah):
        vr2tot=bulgerot(R,mb,bb)+diskrot(R,0,md,ad,bd)+halorot(R,mh,ah)
        return 10*pow(vr2tot,0.5)

#vrinitial=rotcurve(radiusformodel,409,0.23,2856,4.22,0.292,1018,100,2.562)
#plt.plot(radiusformodel, vrinitial, 'k',label="Best Fit (Irrgang 2013)")
#plt.savefig("vrot.pdf")

def residual(params, x, data, uncertainty):
        mb = params['mb']
        bb = params['bb']
        md = params['md']
        ad = params['ad']
        bd = params['bd']
        mh = params['mh']
        ah= params['ah']
        model = rotcurve(x,mb,bb,md,ad,bd,mh,ah)
        return (data-model) / uncertainty

params = Parameters()
params.add('mb', value=175,min=0)
params.add('bb', value=0.184,min=0)
params.add('md', value=2829,min=0)
params.add('ad', value=4.85,min=0)
params.add('bd', value=0.305,min=0)
params.add('mh', value=69725,min=0)
params.add('ah', value=200,min=0)
out = minimize(residual, params, args=(radius,vel, (poser+neger)/2))



#result = gmodel.fit(vel, R=radius, mb=409.0,bb=0.23,md=2856,ad=4.22,bd=0.292,mh=1018,lam=100,ah=2.562)

print(fit_report(out))
file=open('fitreport_model2.dat','w')
file.write(fit_report(out))
file.close()


#################
#plotting:
#################

radiusformodel=np.linspace(0,200.0,1000)
vrinitial=rotcurve(radiusformodel,175,0.184,2829,4.85,0.305,69725,200)
vrmodel=rotcurve(radiusformodel,out.params['mb'].value,out.params['bb'].value,out.params['md'].value,out.params['ad'].value,out.params['bd'].value,out.params['mh'].value,out.params['ah'].value)



                
plt.plot(radiusformodel, vrinitial, 'k',label="Best Fit (Irrgang 2013)")
plt.plot(radiusformodel, vrmodel, 'r', label='Best fit (this work)')
plt.errorbar(radius1, vel1, yerr=(poser1,neger1),fmt='.', label='Eilers 2019')
plt.errorbar(radius2, vel2, yerr=(poser2,neger2),fmt='+', label='Bhattacharjee 2014')
plt.legend()
plt.savefig("Rotcur_ModelII_eil_bhatta.pdf")







'''
=columns_to_keep)
df = pd.read_csv("./clusters_simbad.csv",usecols=columns_to_keep)

#columns_to_keep = ['ra_epoch2000','dec_epoch2000','plx_hip','pmra_hip','pmde_hip','radvel_simbad']#'rv_other']
#dfstar = pd.read_csv("./Hypervel_Hooglist.csv")

Name=df['main_id']
#Name=df['SimbadName']
#orbstar=orbit.Orbit.from_name('HIP 42038')
#number=15

#Namestar="PG 1610+062"

starra=dfstar['ra_epoch2000']
stardec=dfstar['dec_epoch2000']
starplx=dfstar['plx_dr2']
starpmra=dfstar['pmra']
starpmdec=dfstar['pmde']
starrv=dfstar['rv_other']##rv_other']

print(starplx.iloc[number])

#orbstar=orbit.Orbit([starra.iloc[number],stardec.iloc[number],(1/starplx.iloc[number]),starpmra.iloc[number],
#                     starpmdec.iloc[number],starrv.iloc[number]],radec=True)#ro=8.3,vo=229.,solarmotion='schoenrich')
#orbstar.integrate(ts,pot=pott)
starresult=Simbad.query_object("HIP 114569")
c = coord.SkyCoord(starresult['RA'],starresult['DEC'], unit=(units.hourangle, units.deg))
#########################

parallaxgaia=1./3.01
parallaxgaiaoffset=0.0
parallax=parallaxgaia-parallaxgaiaoffset
rv=99.7
pma=45.76
pmd=32.22
ra=c.ra.value[0]
dec=c.dec.value[0]
print(ra)
orbstar=orbit.Orbit([ra,dec,(1.64),pma,pmd,rv],radec=True)#ro=8.3,vo=229.,solarmotion='schoenrich')orbstar.integrate(ts,pot=pott)
orbstar.integrate(ts,pot=pott)
faulty=[256,102,140,257,496,598,732,751,816,817,818,819,820,821,832,833,834,821,822,823,824,825,826,827,828,829,101]
faulty2=[5,6,17,18,29,31,34]
suspects = pd.DataFrame({'Name':[],'Mindist':[],'Time':[]})
flag=0


for i in range(len(Name)):
    j=i
    
    if j not in faulty:
        orb=orbit.Orbit.from_name(Name.iloc[j])
        #orb=orbit.Orbit([157.95,-58.24,6.0,-6.80,
         #            3.47,-31.5],radec=True)
        orb.integrate(ts,pot=pott)
        x1=pow((orb.x(ts)-orbstar.x(ts)),2)
        y1=pow((orb.y(ts)-orbstar.y(ts)),2)
        z1=pow((orb.z(ts)-orbstar.z(ts)),2)
        dist1=x1+y1+z1
        dist2=np.sqrt(dist1)
        mindist=np.min(dist2)
        time=np.argmin(dist2)
        #print(Name.iloc[j],j,ts[time],mindist)
        print(j)
    if(mindist<0.5):
        print(mindist)
        print(Name.iloc[j])
        data = [{'Name':Name.iloc[j],'Mindist':mindist,'Time':ts[time]}]
        suspects.loc[flag]=list(data[0].values())
        print(suspects)
        flag=flag+1
        #plt.plot(orb.x(ts),orb.y(ts),'ro')
        #plt.plot([-3.1],[-5.5],marker='o',label='point')
        #plt.plot(orbstar.x(ts),orbstar.y(ts))

        #plt.legend()
        #plt.show()
suspects=suspects.sort_values(by=['Mindist'],ascending=True)
print(suspects)


#######Associations from Gaia:

columns_to_keep = ['Name','Alpha','Delta','Plx','PMA','PMD','RV']#'rv_other']
df2 = pd.read_csv("./New_Associations.csv",usecols=columns_to_keep)
Nameassoc=df2['Name']
assocra=df2['Alpha']
assocdec=df2['Delta']
assocplx=df2['Plx']
assocpmra=df2['PMA']
assocpmdec=df2['PMD']
assocrv=df2['RV']
for i in range(len(Nameassoc)):
    j=i
    
    if j not in faulty:
        orbassoc=orbit.Orbit([assocra.iloc[i],assocdec.iloc[i],(1/assocplx.iloc[i]),assocpmra.iloc[i],
                     assocpmdec.iloc[i],assocrv.iloc[i]],radec=True)#ro=8.3,vo=229.,solarmotion='schoenrich')
        orbassoc.integrate(ts,pot=pott)
        x1=pow((orbassoc.x(ts)-orbstar.x(ts)),2)
        y1=pow((orbassoc.y(ts)-orbstar.y(ts)),2)
        z1=pow((orbassoc.z(ts)-orbstar.z(ts)),2)
        dist1=x1+y1+z1
        dist2=np.sqrt(dist1)
        mindist=np.min(dist2)
        time=np.argmin(dist2)
        print(Nameassoc.iloc[j],j,ts[time],mindist)
    if(mindist<0.05):
        print(mindist)
        print(Nameassoc.iloc[j])
        data = [{'Name':Nameassoc.iloc[j],'Mindist':mindist,'Time':ts[time]}]
        suspects.loc[flag]=list(data[0].values())
        print(suspects)
        flag=flag+1
        plt.plot(orb.x(ts),orb.y(ts),'ro')
        #plt.plot([-3.1],[-5.5],marker='o',label='point')
        #plt.plot(orbstar.x(ts),orbstar.y(ts))

        #plt.legend()
        plt.show()
print(suspects)
'''
