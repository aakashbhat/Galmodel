import numpy as np
import math
from scipy.integrate import dblquad


#Potential for Model I of Irrgang 2013 in terms of r and z components:
def modelI_pott(r,z=0,mb=409,bb=0.23,md=2856,ad=4.22,bd=0.292,mh=1018,ah=2.562,lamda=200):
    #if hasattr(r, "__len__") and hasattr(N, "__len__"):
    #    radius=np.meshgrid(r,z)
    #else:
    rr,zz=np.meshgrid(r,z)
    radius=np.sqrt(rr**2 + zz**2)

    bulge=-mb/np.sqrt(radius**2+bb**2)
    disk=-md/np.sqrt(rr**2+np.square(ad+np.sqrt(zz**2+bd**2)))
    
    halo=radius

    for i in range(len(rr)):
        for j in range(len(zz)):
            if radius[i][j]<lamda:
                halo[i][j]=(mh/ah)*(np.log((ah+radius[i][j])/(ah+lamda))-(lamda/(ah+lamda)))
            else:
                halo[i][j]=-mh*np.square(lamda)/(radius[i][j]*ah*(ah+lamda))
    return bulge+disk+halo

#Potential of Model III of Irrgang 2013 in terms of r and z:
def modelIII_pott(r,z=0,mb=439,bb=0.236,md=3096,ad=3.262,bd=0.289,mh=142200,ah=45.02):
    radius=r*r+z*z
    bulge=-1*mb/np.sqrt(radius*radius+bb*bb)
    disk=-1*md/np.sqrt(r*r+np.square(ad+np.sqrt(z*z+bd*bd)))
    halo=-1*mh*np.log((ah+radius)/ah)/radius
    return bulge+disk+halo

###Vescape for Models I and III but in x,y,z coordinates at the moment:
def vescape(radius,z,model=1,mb=0,bb=0,md=0,ad=0,bd=0,mh=0,ah=0,lamda=0):
    
    if model==1:
        if mb!=0:
            pott=modelI_pott(radius,z,mb,bb,md,ad,bd,mh,ah,lamda)
        else:
            pott=modelI_pott(radius,z)
    elif model==3:
        if mb!=0:
            pott=modelIII_pott(radius,z,mb,bb,md,ad,bd,mh,ah)
        else:
            pott=modelIII_pott(radius,z)
    else: pott==0

    vesc=np.sqrt(-2*pott)
    return 10*vesc


####Miyamoto Nagai density calculation:
def bulge_density(x,y,z,mb=409,bb=0.23):
    xx,yy,zz=np.meshgrid(x,y,z)
    radius=np.sqrt(xx**2 + yy**2+zz**2)
    density=3*bb*bb*mb/(4*math.pi*np.power((radius**2+bb**2),2.5))
    return density
#####Disk density calculation:
def disk_density(x,y,z,md=2856,ad=4.22,bd=0.292):
    constant=bd*bd*md/(4*math.pi)
    xx,yy,zz=np.meshgrid(x,y,z)
    radius=xx**2 + yy**2
    extraz=np.square(ad+np.sqrt(zz*zz+bd*bd))
    num=ad*radius+(ad+3*np.sqrt(bd*bd+zz*zz))*np.square(ad+np.sqrt(bd**2+zz**2))
    den=np.power((zz**2+bd*bd),1.5)*np.power((radius+extraz),2.5)
    density=constant*num/den
    return density

#####Halo density for Alan Santillan model used in model I:
def halo_as_density(x,y,z,mh=1018,ah=2.562,lamda=200):
    
    constant=mh/(4*math.pi*ah)
    xx,yy,zz=np.meshgrid(x,y,z)
    radius=np.sqrt(xx**2 + yy**2+zz**2)
    
    num=radius+2*ah
    den=radius*np.square(ah+radius)
    density=np.where(radius<lamda,constant*num/den,0)
    return density

#####Halo density for Model III:

def halo_NFW_density(x,y,z,mh=142200,ah=45.02):
    constant=mh/(4*math.pi)
    xx,yy,zz=np.meshgrid(x,y,z)
    radius=np.sqrt(xx**2 + yy**2+zz**2)
    den=radius*np.square(ah+radius)
    density=constant/den
    return density

#####This function calculates the velocity in the galactocentric frame of galpy:
####To calculate space velocity one needs to take into account the rotation of the sun as well
def grf_velocity(name,parallax,pmra,pmdec,rv):
    from astroquery.simbad import Simbad
    customSimbad = Simbad()
    customSimbad.add_votable_fields('ra(opt)','dec(opt)')
    result=customSimbad.query_object(name)
    from galpy.util.coords import radec_to_lb, vrpmllpmbb_to_vxvyvz,pmrapmdec_to_pmllpmbb,lb_to_radec
    l=175.98
    b=47.051#radec_to_lb(ra, dec, degree=True, epoch=2000.0)
    ra,dec=lb_to_radec(l, b, degree=True, epoch=2000.0)
    pmll,pmbb=pmrapmdec_to_pmllpmbb(pmra, pmdec,ra,dec,degree=True, epoch=2000.0)
    vx,vy,vz=vrpmllpmbb_to_vxvyvz(rv, pmll, pmbb, l, b, 1/parallax, XYZ=False, degree=True)
    print(np.sqrt(vx**2+vy**2+vz**2))
    return vx,vy,vz

###Integration of densities to get masses within a sphere of radius R:

def bulge_mass(R,mb,bb):
    area = dblquad(lambda z, r: 3*r*bb*bb*mb/(pow((r**2+z**2+bb**2),2.5)), 0, R, lambda r: 0, lambda r: np.sqrt(R*R-r*r))
    return area

def disk_mass(R,md,ad,bd):
    const=bd**2*md
    area = dblquad(lambda z, r: r*const*(ad*r**2+(ad+3*np.sqrt(bd**2+z**2))*np.square(ad+np.sqrt(bd**2+z**2)))/(pow(z**2+bd**2,1.5)*pow((r**2+(ad+np.sqrt(z**2+bd**2))**2),2.5)), 0, R, lambda r: 0, lambda r: np.sqrt(R*R-r*r))
    return area

def halo_as_mass(R,mh,ah,lamda):
    mass=0
    if R<lamda:
        mass+=mh*R**2/(ah*(ah+R))
    else:
        mass=mh*lamda**2/(ah*(ah+lamda))
    return mass

def halo_NFW_mass(R,mh,ah):
    area = dblquad(lambda z, r: r*mh/(np.sqrt(r**2+z**2)*np.square(ah+np.sqrt(r**2+z**2))), 0, R, lambda r: 0, lambda r: np.sqrt(R*R-r*r))
    return area



#####Calculate A:
def omega_bulge_miyamoto(R,mb=409,bb=0.23):
    return 10*np.sqrt(mb/np.power(R**2+bb**2,1.5))

def omega_disk_miyamoto(R,md=2856,ad=4.22,bd=0.292):
    return 10*np.sqrt(md/np.power(R**2+(ad+bd)**2,1.5))

def omega_as_halo(R,mh=1018,ah=2.562,lamda=200):
    if R<lamda:
        return 10*np.sqrt(mh/(ah*(ah+R)*R)) 
def omega_nfw_halo(R,mh=142200,ah=45.02):
    r2=R+ah
    omega=mh*((np.log(r2/ah)/R)-(1/r2))
    return 10*np.sqrt(omega/R**2)
    
def omega(R,Model=1,mb=409,bb=0.23,md=2856,ad=4.22,bd=0.292,mh=1018,ah=2.562):
    if Model==1:
        omega=np.sqrt(np.square(omega_bulge_miyamoto(R,mb,bb))+np.square(omega_disk_miyamoto(R,md,ad,bd))+np.square(omega_as_halo(R,mh,ah)))
    elif Model==3:
        mb=439
        bb=0.236
        md=3096
        bd=0.289
        ad=3.262
        mh=142200
        ah=45.02
        omega=np.sqrt(np.square(omega_bulge_miyamoto(R,mb,bb))+np.square(omega_disk_miyamoto(R,md,ad,bd))+np.square(omega_nfw_halo(R,mh,ah)))
    return omega

def Oort(Rsun=8.4,Model=1,mb=409,bb=0.23,md=2856,ad=4.22,bd=0.292,mh=1018,ah=2.562):
    #####Formula for derivative of omega = d(omega^2)/dR*2omega
    der_omega2_bulge=-3*mb*Rsun/pow(Rsun**2+bb**2,2.5)
    der_omega2_disk=-3*md*Rsun/np.power(Rsun**2+(ad+bd)**2,2.5)
    r2=Rsun+ah

    if Model==1:
        der_omega2_as_halo=-mh*(ah+2*Rsun)/(ah*Rsun**2*(r2)**2)
        omega_total=omega(Rsun,Model)
        der_omega2=der_omega2_as_halo+der_omega2_bulge+der_omega2_disk
        A_oort=-0.5*Rsun*der_omega2/omega_total
    elif Model==3:
        
        der_omega2_nfw_halo=(mh/Rsun**4)*((Rsun/r2)-3*np.log(r2/ah)+((3*Rsun**2+2*Rsun*ah)/r2**2))
        omega_total=omega(Rsun,Model)
        der_omega2=der_omega2_nfw_halo+der_omega2_bulge+der_omega2_disk
        A_oort=-0.5*Rsun*der_omega2/omega_total
    
    return A_oort*50,(A_oort*50-omega_total)



    
