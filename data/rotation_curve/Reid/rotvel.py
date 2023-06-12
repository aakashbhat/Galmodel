
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

U_lsr,V_lsr,W_lsr=11.1, 12.24, 7.25
Ue,Ve,We = 1.69, 2.47, 0.87
vsun = 229
vsune= 0.2
rsun= 8.178
rsune=0.035
def your_func(alpha,delta,plx,pma,pmd,vr,splx,spma,spmd,svr):
    s=10^6
    pmra=np.random.normal(loc=pma,scale=spma,size=s)
    pmdec=np.random.normal(loc=pmd,scale=spmd,size=s)
    distmin=1./(plx+splx)
    distmax=1./(plx-splx)
    dister=(distmax-distmin)
    dist=np.random.normal(loc=1.0/plx,scale=dister/2,size=s)
    rad=np.random.normal(loc=vr,scale=svr,size=s)
    U=np.random.normal(loc=U_lsr,scale=Ue,size=s)
    V=np.random.normal(loc=V_lsr,scale=Ve,size=s)
    W=np.random.normal(loc=W_lsr,scale=We,size=s)
    Vo=np.random.normal(loc=vsun,scale=vsune,size=s)
    Rsun=np.random.normal(loc=rsun,scale=rsune,size=s)
    x=[]
    y=[]
    z=[]
    vx=[]
    vy=[]
    vz=[]
    vr=[]
    vphi=[]


    # Coordinates in Galactic frame:
    for i in range(s):
        g = coord.LSR(ra = alpha*u.degree, dec = delta*u.degree, distance =dist[i]*u.kpc,pm_ra_cosdec=pmra[i]*u.mas/u.yr, pm_dec = pmdec[i]*u.mas/u.yr,radial_velocity=rad[i]*u.km/u.s,v_bary=(U_lsr,V_lsr,W_lsr))
        # convert to ICRS:
        c = g.transform_to(coord.Galactocentric(galcen_distance=Rsun[i]*u.kpc,galcen_v_sun=(U[i],Vo[i]+V[i],W[i])*u.km/u.s))
        
        x.append(c.x/u.kpc)
        y.append(c.y/u.kpc)
        z.append(c.z/u.kpc)
        vx.append(c.v_x*u.s/u.km)
        vy.append(c.v_y*u.s/u.km)
        vz.append(c.v_z*u.s/u.km)
        vr.append((c.x*c.v_x+c.y*c.v_y)*u.s/(np.sqrt(c.x*c.x+c.y*c.y)*u.km))
        vphi.append((c.y*c.v_x-c.x*c.v_y)*u.s/(np.sqrt(c.x*c.x+c.y*c.y)*u.km))


  
    return np.mean(x),np.mean(y),np.mean(z),np.mean(vx),np.mean(vy),np.mean(vz),np.mean(vr),np.mean(vphi),np.std(x),np.std(y),np.std(z),np.std(vx),np.std(vy),np.std(vz),np.std(vr),np.std(vphi)

columns_to_keep = ['ra','dec', 'plx','e_plx','pmX','e_pmX','pmY','e_pmY','VLSR','e_VLSR']
df = pd.read_csv("Reid/reid_data_2019_finalcut.csv")# usecols=columns_to_keep,decimal='.')
    #df = df[df.nmu >1]
ra=df['ra']
dec=df['dec']
pma=df['pmX']
pmd=df['pmY']
spma=df['e_pmX']
spmd=df['e_pmY']
plx=df['plx']
splx=df['e_plx']
rv=df['VLSR']
srv=df['e_VLSR']

#Name=df['Associa']
#vr=df['Vr']
#svr=df['s_VR']
#r=df['r']
df2 = pd.DataFrame({'ra':[],'x':[],'y':[],'z':[],'vx':[],'vy':[],'vz':[],'vr':[],'vphi':[],'ex':[],'ey':[],'ez':[],'evx':[],'evy':[],'evz':[],'evr':[],'evphi':[],'r':[]})

for i in range(len(ra)):
    print(i)
    right=ra.iloc[i]
    a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p=your_func(ra.iloc[i],dec.iloc[i],plx.iloc[i],pma.iloc[i],pmd.iloc[i],rv.iloc[i],splx.iloc[i],spma.iloc[i],spmd.iloc[i],srv.iloc[i])
    data = [{'ra':right,'x':a,'y':b,'z':c,'vx':d,'vy':e,'vz':f,'vr':g,'vphi':h,'ex':i,'ey':j,'ez':k,'evx':l,'evy':m,'evz':n,'evr':o,'evphi':p,'r':np.sqrt(a*a+b*b)}]
    df2.loc[i]=list(data[0].values())
    
df2.to_csv('reid_rotationcurve_2019_ourparams.csv')
plt.errorbar(df2['r'],df2['vphi'],yerr=df2['evphi'],fmt='r+',ecolor='r')
plt.savefig('reid_rotcurve_2019_ourparams.pdf')
kutta
    
dataframe['PMA','S_PMA','PMD','S_PMD','Alpha','Delta'] = dataframe.apply(your_func, axis=1)

print(dataframe)
kutat
s=10^6
pml=np.random.normal(loc=-32.1,scale=0.1,size=s)
pmb=np.random.normal(loc=-13.1,scale=0.1,size=s)
l1=298.5171
b1=5.4934
pmalpha=[]
pmdelta=[]
# Coordinates in Galactic frame:
for i in range(s):
    g = coord.Galactic(l = l1*u.degree, b = b1*u.degree, pm_l_cosb =pml[i]*u.mas/u.yr, pm_b = pmb[i]*u.mas/u.yr)
    # convert to ICRS:
    c = g.transform_to(coord.ICRS)
    pma=c.pm_ra_cosdec*u.yr/u.mas
    pmd=c.pm_dec*u.yr/u.mas
    pmalpha.append(pma.value)
    pmdelta.append(pmd.value)

print("PMAlpha*: {} +/- {}".format(np.mean(pmalpha),np.std(pmalpha)))
print("PMDelta: {} +/- {}".format(np.mean(pmdelta),np.std(pmdelta)))
print("Alpha:",c.ra.value)
print("Delta:",c.dec.value)
