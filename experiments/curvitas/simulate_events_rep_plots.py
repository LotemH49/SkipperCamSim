from pathlib import Path

import numpy as np
import sep
from matplotlib.patches import Ellipse
from clean_masking_functions import *
import matplotlib.pyplot as plt
from mpmath import hyp1f1
import tqdm

_CURVITAS_DIR = Path(__file__).resolve().parent
img_str = _CURVITAS_DIR / "object_luminosity.fits"

with fits.open(img_str) as hdul:
    data = hdul[0].data.astype(np.float32)  # Convert to float for sep

Expo=2#s
t=Expo*np.arange(len(data[:,0]))
#flux_matrix[:,i] get the intensity curve of the ith event

plt.imshow(data,origin='lower')
plt.show()

rand_objt=np.random.randint(0,len(data[0,:]))
random_time=np.random.randint(0,len(data[:,0]))*Expo#in seconds since time of observation

## Black hole parameters

def magnificacion_villa(t, t0, t_E, u0):
    u = np.sqrt(u0**2 + ((t - t0) / t_E)**2)
    A = (u**2 + 2) / (u * np.sqrt(u**2 + 4))
    return A,u


def get_wave_parameter(wavelen,M):
    return 5.98*(M/1e-10)*(wavelen/6210) #M in solar mass, wavelen in Armstrong


def get_wave_magnification(t, wave_par, t0, t_E, u0):
    u = np.sqrt(u0**2 + ((t - t0) / t_E)**2)
    z = 0.5j * wave_par * u**2
    
    
    Prefactor = (np.pi * wave_par) / (1 - np.exp(-np.pi * wave_par))

    F = np.zeros(len(u), dtype=complex)
    for i in range(len(u)):
        a = 0.5j * wave_par
        F[i] = complex(hyp1f1(a, 1, z[i]))  #Es un mpmath object lo paso a complex de python

    return Prefactor * np.abs(F)**2  , u


def get_one_amp(t, wave_par, t0, t_E, u0):
    u = np.sqrt(u0**2 + ((t - t0) / t_E)**2)
    z = 0.5j * wave_par * u**2
    
    
    Prefactor = (np.pi * wave_par) / (1 - np.exp(-np.pi * wave_par))

    a = 0.5j * wave_par
    F = complex(hyp1f1(a, 1, z))  #Es un mpmath object lo paso a complex de python

    return Prefactor * np.abs(F)**2 


def finite_source_wave_mag(t, wave_par, rho_star, N=500):
    u = np.sqrt(u0**2 + ((t - t0) / t_E)**2)


    #Armas un circulo tamanio N (no importa tanto cuanto es la grilla da un poco mas de acc)
    r = np.linspace(0, rho_star, int(np.sqrt(N)))
    theta = np.linspace(0, 2*np.pi, int(np.sqrt(N)))
    rr, tt = np.meshgrid(r, theta)
    
    # en polares
    x = rr * np.cos(tt)
    y = rr * np.sin(tt)

    # Compute radial offset from lens to each source point
    d = np.sqrt((u - x)**2 + y**2)

    # magnificacion puntual en cada punto con wave optiks
    A  = get_one_amp(t,wave_parameter,t0,t_E,u)

    # integras (suma en el circulo)
    dA = (rho_star / len(r)) * (2 * np.pi / len(theta))
    integral = np.sum(A * rr) * dA

    # Normalizar por pi * rho_star^2
    A_finite = integral / (np.pi * rho_star**2)

    return A_finite,u


rho_estrella=1

# Parameters
u0 = 1e-3
t0 = random_time 

Mass= 1e-10 #@in solar mass units

wavelen_ble=6000 #Armstrong

blk_hole_time=63*86400*np.sqrt(Mass) #in seconds

t_E =blk_hole_time

print("Event time: "+str(blk_hole_time))
wave_parameter=get_wave_parameter(wavelen_ble,Mass)

A,impacts=get_wave_magnification(t,wave_parameter,t0,t_E,u0)


A_villa,u_villa = magnificacion_villa(t, t0, t_E, u0)

plt.plot(impacts,A,label='Wave optics')
plt.plot(u_villa,A_villa,label='No wave optics')
plt.show()

#######3

A_f = [finite_source_wave_mag(t_tiempito, wave_parameter, rho_estrella,500)[0] for t_tiempito in tqdm.tqdm(t)]

u_f =[finite_source_wave_mag(t_tiempito, wave_parameter, rho_estrella,500)[1] for t_tiempito in tqdm.tqdm(t)]

#plt.


plt.plot(impacts,A,label='Wave optics')
plt.plot(u_f,A_f,label='TODO')
plt.legend()
plt.show()


plt.errorbar(t,A_f*data[:,rand_objt],yerr=np.sqrt(A_f*data[:,rand_objt]),fmt='o',alpha=0.7,color='red')
plt.xlabel("Time")
plt.ylabel("Magnification")

plt.axvline(t0, color='gray', linestyle='--', label='Peak time')
plt.grid(True)
plt.legend()
plt.show()

def get_one_amp2(wave_par,u):
    z = 0.5j * wave_par * u**2
    

    Prefactor = (np.pi * wave_par) / (1 - np.exp(-np.pi * wave_par))

    a = 0.5j * wave_par
    F = complex(hyp1f1(a, 1, z))  #Es un mpmath object lo paso a complex de python

    return Prefactor * np.abs(F)**2 

def finite_source_wave_mag(u, wave_par, rho_star, N=2800):
    # Create polar grid over circular disk of radius rho_star
    r = np.linspace(0, rho_star, int(np.sqrt(N)))
    theta = np.linspace(0, 2*np.pi, int(np.sqrt(N)))
    rr, tt = np.meshgrid(r, theta)
    
    # Coordinates in source plane
    x = rr * np.cos(tt)
    y = rr * np.sin(tt)

    # Compute radial offset from lens to each source point
    d = np.sqrt((u - x)**2 + y**2)

    # Compute magnification at each point
    A = np.vectorize(lambda u_val: get_one_amp2(wave_par, float(u_val)))(d)

    # Integrate over the disk (area element in polar coords is r * dr * dtheta)
    weights = rr  # Jacobian for polar area element
    dA = (rho_star / len(r)) * (2 * np.pi / len(theta))
    integral = np.sum(A * weights) * dA

    # Normalize by pi * rho_star^2
    A_finite = integral / (np.pi * rho_star**2)
    return A_finite


Mass= 1e-10 #in solar mass units

u0 =1e-3

u_values = np.sqrt(u0**2 + ((t - t0) / t_E)**2)

plt.plot(impacts, A,ls='dashed',color='k', label='Point source')
for rho_star in tqdm.tqdm([0.1,0.5,1]):
    A_f = [finite_source_wave_mag(u, wave_parameter, rho_star, N=1000) for u in tqdm.tqdm(u_values)]
    plt.plot(u_values, A_f, label=f'Finite source (ρ*={rho_star})')

plt.xlabel('u')
plt.ylabel('Magnification')
plt.legend()
plt.grid()
plt.title('Wave Optics Lensing with Finite Source Size')
plt.show()