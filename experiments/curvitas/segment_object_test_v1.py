from pathlib import Path

import numpy as np
import sep
from matplotlib.patches import Ellipse
from clean_masking_functions import *
import fitsio
import matplotlib.pyplot as plt
from matplotlib import rcParams
from astropy.io import fits
rcParams['figure.figsize'] = [10., 8.]

_CURVITAS_DIR = Path(__file__).resolve().parent
_SIM_IMAGES_DIR = _CURVITAS_DIR / "simulated_images"


def add_dc_to_img(img,Dc_events,num_pixels,overscan_size):
    imagen_dc = np.zeros(num_pixels)  
    uni_int_1=np.random.randint(0,num_pixels[0],size=(int(round(Dc_events))))
    uni_int_2=np.random.randint(7,num_pixels[1]-overscan_size,size=(int(round(Dc_events))))#prescan_hardcoded
    for i in range(int(round(Dc_events))):
        imagen_dc[uni_int_1[i],uni_int_2[i]]+=1
    return img+imagen_dc

def add_read_to_img(img,readout_nos,num_pixels):
    read_out_img=np.random.normal(0,scale=readout_nos,size=(num_pixels))
    return img+read_out_img

def simulate_img(img_stars,dc_events,read_out_noise,overscan_size):
    _zeros=img_stars
    num_pix=np.shape(_zeros)
    _imagen_dc=add_dc_to_img(_zeros,dc_events,num_pix,overscan_size)
    _imagen_read=add_read_to_img(_zeros,read_out_noise,num_pix)

    return _imagen_dc+_imagen_read



def simulate_one_microchip(decam_img,gainA,gainB,Exposure,readout_noise):
    Expo_time_fromhead=201.1361098#seconds

    #substract OS
    last_col_A=1079
    first_col_B=1080
    data[:, :last_col_A] -= np.median(data[:,8:54],axis=1)[:, np.newaxis]
    data[:, first_col_B:] -= np.median(data[:,2105:-7],axis=1)[:, np.newaxis]
    ## To electrons:
    data[:, :last_col_A]=data[:, :last_col_A]*float(gainA)
    data[:, first_col_B:]=data[:, first_col_B:]*float(gainB)
    
    bkg = sep.Background(data,  bw=64, bh=64, fw=3, fh=3)
    data_sub = data - bkg
    
    data_sub_scaled_background=data/Expo_time_fromhead
    data_sub_scale=(data-bkg)/Expo_time_fromhead

    bkg = sep.Background(data_sub_scaled_background,  bw=64, bh=64, fw=3, fh=3)
    
    mu_for_sim=bkg.globalrms**2

    new_img_test=np.zeros(shape=(700,650))

    os_size=27
    microchip_size=(700,650-os_size)

    new_slice_for_array=data_sub_scale[200:200+microchip_size[0],907:900+microchip_size[1]]


    new_img_test[:,7:microchip_size[1]]+=new_slice_for_array


    poisson_events=int(Exposure*mu_for_sim*microchip_size[0]*microchip_size[1])
    return simulate_img(new_img_test,poisson_events,readout_noise,os_size)

def simulate_stack_microchip(decam_img,gainA,gainB,Exposure,readout_noise,N_imgs):
    Expo_time_fromhead=201.1361098#seconds

    #substract OS
    last_col_A=1079
    first_col_B=1080
    data[:, :last_col_A] -= np.median(data[:,8:54],axis=1)[:, np.newaxis]
    data[:, first_col_B:] -= np.median(data[:,2105:-7],axis=1)[:, np.newaxis]
    ## To electrons:
    data[:, :last_col_A]=data[:, :last_col_A]*float(gainA)
    data[:, first_col_B:]=data[:, first_col_B:]*float(gainB)
    #substract background to get "only stars"
    bkg = sep.Background(data,  bw=64, bh=64, fw=3, fh=3)
    data_sub = data - bkg
    #sacel by exposure
    data_sub_scaled_background=data/Expo_time_fromhead
    data_sub_scale=data_sub/Expo_time_fromhead
    #get sky noise stimation
    bkg = sep.Background(data_sub_scaled_background,  bw=64, bh=64, fw=3, fh=3)
    #mean is this value squared 
    mu_for_sim=bkg.globalrms**2

    new_img_test=np.zeros(shape=(700,650))

    os_size=27
    microchip_size=(700,650-os_size)

    new_slice_for_array=data_sub_scale[200:200+microchip_size[0],907:900+microchip_size[1]]
    
    new_slice_for_array[new_slice_for_array < 0] = 0

    poisson_image = np.random.poisson(new_slice_for_array)

    new_img_test[:,7:microchip_size[1]]+=poisson_image

    eventos_only=new_img_test>mu_for_sim*5.5

    #poisson_events=int(Exposure*mu_for_sim*microchip_size[0]*microchip_size[1])
    count_expo=0
    for i in range(N_imgs):
        count_expo+=2.5
        _SIM_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        output_filename = _SIM_IMAGES_DIR / f"image_sim_{count_expo}.fits"
        array_2d=simulate_img(new_img_test,0,readout_noise,os_size)
        hdu = fits.PrimaryHDU(array_2d)
        hdu.writeto(output_filename, overwrite=True)
    return None


img_str = _CURVITAS_DIR / "data_ext8.fits"
with fits.open(img_str) as hdul:
    data = hdul[1].data.astype(np.float32)  # Convert to float for sep
    header =hdul[1].header

readout_noise=10 #electrons
exposure_single=2.5 #seconds
simulate_stack_microchip(data,header['GAINA'],header['GAINB'],exposure_single,readout_noise,14400)
#plt.hist(flatten(simulate_one_microchip(data,header['GAINA'],header['GAINB'],exposure_single,readout_noise)),bins=np.linspace(0,100,1000),histtype='step')
#plt.show()