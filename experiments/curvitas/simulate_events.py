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

def magnificacion_villa(t, t0, t_E, u0):
    u = np.sqrt(u0**2 + ((t - t0) / t_E)**2)
    A = (u**2 + 2) / (u * np.sqrt(u**2 + 4))
    return A,u

def get_wave_parameter(wavelen,M):
    return 5.98*(M/1e-10)*(wavelen/6210) #M in solar mass, wavelen in Armstrong

def get_one_amp2(wave_par,u):
    z = 0.5j * wave_par * u**2
    

    Prefactor = (np.pi * wave_par) / (1 - np.exp(-np.pi * wave_par))

    a = 0.5j * wave_par
    F = complex(hyp1f1(a, 1, z))  #Es un mpmath object lo paso a complex de python

    return Prefactor * np.abs(F)**2 

def finite_source_wave_mag(u, wave_par, rho_star, N=2800):
    # Hacemos la grilla de r y theta para integrar
    r = np.linspace(0, rho_star, int(np.sqrt(N)))
    theta = np.linspace(0, 2*np.pi, int(np.sqrt(N)))
    rr, tt = np.meshgrid(r, theta)
    # meshgrid en polares 
    x = rr * np.cos(tt)
    y = rr * np.sin(tt)
    # el offset de la lente a cada punto
    d = np.sqrt((u - x)**2 + y**2)
    # Calcula la magnificacion puntual de cada punto del circulo
    A = np.vectorize(lambda u_val: get_one_amp2(wave_par, float(u_val)))(d)
    # sumo sobre el disco
    weights = rr  # Jacobiano  polares
    dA = (rho_star / len(r)) * (2 * np.pi / len(theta)) #diferenciales de angulo y r
    integral = np.sum(A * weights) * dA
    # Normalizo por pi * rho_star^2
    A_finite = integral / (np.pi * rho_star**2)
    return A_finite

Expo=2#s
#flux_matrix[:,i] get the intensity curve of the ith event
#
plt.imshow(data,origin='lower')
plt.show()

rand_objt=np.random.randint(0,len(data[0,:]))
random_time=np.random.randint(0,len(data[:,0]))*Expo#in seconds since time of observation

#parameters
t=Expo*np.arange(len(data[:,0]))

t0 = random_time 
u0 =1
Mass= 1e-12 #in solar mass units
rho_star=2
blk_hole_time=63*86400*np.sqrt(Mass) #in seconds formula rancia de un paper para calculo rapido REEMPLAZAR POR UNA SERIA
t_E =blk_hole_time

u_values = np.sqrt(u0**2 + ((t - t0) / t_E)**2) #acercamientos de la lente para el calculo de la magnificacion

wavelen_ble=3500 #Armstrong supuestamente esto vamos a usar



print("Event time: "+str(blk_hole_time))
wave_parameter=get_wave_parameter(wavelen_ble,Mass)


A_f = [finite_source_wave_mag(u, wave_parameter, rho_star, N=500) for u in tqdm.tqdm(u_values)]
plt.errorbar(t,A_f*data[:,rand_objt],yerr=np.sqrt(A_f*data[:,rand_objt]),fmt='o',alpha=0.7,color='red')
plt.xlabel("Time")
plt.ylabel("Magnification")
plt.axvline(t0, color='gray', linestyle='--', label='Peak time')
plt.grid(True)
plt.legend()
plt.show()