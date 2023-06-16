import matplotlib.pyplot as plt
import numpy as np
from functions import Oort,halo_NFW_mass,bulge_mass, disk_mass,halo_as_mass,grf_velocity,vescape,modelI_pott,bulge_density,disk_density,halo_as_density,halo_NFW_density
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from typing import NamedTuple
x=np.linspace(0,100,100)
y=np.linspace(-10,10,100)
z=np.linspace(0,100,100)

to_plot="vescape"
model=1
class params(NamedTuple):
    mb: float
    bb: float
    md: float
    ad: float
    bd: float
    mh: float
    ah: float
    


if model==1:
    model_params=params(409,0.23,2856,4.22,0.23,1018,2.562)
elif model==3:
    model_params=params(439,0.236,3096,3.262,0.289,142200,45.02)


if to_plot=="vescape":

    vesc=vescape(x,z,model)    
    ###For Plotting differences:
    vesc_new=vescape(x,z,1,395,0.19,3445,5.11,0.44,510,1.75,53.41)

    fig,ax=plt.subplots()
    im=ax.pcolormesh(x,z,(np.array(vesc)-np.array(vesc_new)))
    cbr=plt.colorbar(im,fraction=0.1)
    plt.xlabel("R (kpc)")
    plt.ylabel("Z (kpc)")
    cbr.set_label(r'Vesc(km$s^{-1}$)')
    plt.savefig("Plots/Escape_velocity/%skpc_difference_model%s_old_new.pdf"%(int(np.max(x)),model))

elif to_plot=="density":
    if model==1:
        diskden=disk_density(x,y,z)
        bulgeden=bulge_density(x,y,z)
        haloden=halo_as_density(x,y,z)
        den=diskden+bulgeden+haloden
    elif model==3:
        diskden=disk_density(x,y,z,md=3096,ad=3.262,bd=0.289)
        bulgeden=bulge_density(x,y,z,mb=439,bb=0.236)
        haloden=halo_NFW_density(x,y,z)
        den=diskden+bulgeden+haloden
    cons=1.0/(10*4.3009)
    
    
    fig,ax=plt.subplots()
    im=ax.pcolormesh(x,z,np.log(den[:,50,:]*cons),cmap="autumn")
    plt.colorbar(im)
    plt.xlabel("X (kpc)")
    plt.ylabel("Z (kpc)")
    #plt.savefig("Plots/Density/density_Model{}_x_z_10kpc.pdf".format(model))
    plt.show()
    '''
    #plt.savefig("Plots/Density/disk_x_y.pdf")
    fig2,ax2=plt.subplots()
    im2=ax2.pcolormesh(x,z,np.log(den[:,11,:]))
    plt.colorbar(im2)
    plt.xlabel("X (kpc)")
    plt.ylabel("Z (kpc)")
    #plt.show()
    plt.savefig("Plots/Density/density_x_z_10kpc.pdf")
    '''

elif to_plot=="grf":
    from astroquery.simbad import Simbad
    print(Simbad.get_field_description ('ra(opt)'))
    vx,vy,vz=grf_velocity("us 708",0.067,-5.44,1.776,708)
    print(vx,vy,vz)

elif to_plot=="mass":
    if model==1: 
        bulgemass=bulge_mass(200,409,0.23)
        diskmass=disk_mass(200,2856,4.22,0.292)
        halo_as_mass=halo_as_mass(200,1018,2.562,200)
        total_mass=bulgemass[0]+diskmass[0]+halo_as_mass

    elif model==3:
        bulgemass=bulge_mass(200,439,0.236)
        diskmass=disk_mass(200,3096,3.262,0.289)
        halo_NFW_mass=halo_NFW_mass(200,142200,45.02)
        total_mass=bulgemass[0]+diskmass[0]+halo_NFW_mass[0]

    print("In Galactic Masses: ",total_mass)
    print("In solar Masses: ",total_mass*2.325e-5)
    #print("In Galactic Masses: ",diskmass)
    #print("In solar Masses: ",diskmass[0]*2.325e-5)

elif to_plot=="Oort":
    A,B=Oort(8.33,model,model_params.mb,model_params.bb,model_params.md,model_params.ad,model_params.bd,model_params.mh,model_params.ah)
    
    print(A,B)

    
