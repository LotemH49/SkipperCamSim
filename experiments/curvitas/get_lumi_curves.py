from glob import glob
from pathlib import Path

import numpy as np
import sep
from matplotlib.patches import Ellipse
from clean_masking_functions import *
import matplotlib.pyplot as plt
from matplotlib import rcParams
from astropy.io import fits
import tqdm

_CURVITAS_DIR = Path(__file__).resolve().parent
folder_with_fits = str(_CURVITAS_DIR / "simulated_images")
lista_archivos = sorted(glob(folder_with_fits + "/*.fits"))    


with fits.open(lista_archivos[0]) as hdul:
    data = hdul[0].data.astype(np.float32)  # Convert to float for sep

bkg = sep.Background(data,  bw=64, bh=64, fw=3, fh=3)

data_sub = data - bkg

objects =  sep.extract(data_sub, 3, err=bkg.globalrms)

kron_radii,_ = sep.kron_radius(data_sub, objects['x'], objects['y'],objects['a'], objects['b'], objects['theta'],6.0)  # 6 is the Kron factor (default in SExtractor)

# Define aperture radius as a multiple of Kron radius (e.g., 2.5×)
r_apertures = 2.5 * np.array(kron_radii)
n_objects = len(objects)
flux_matrix = np.zeros((len(lista_archivos), n_objects))
print(n_objects)

for i, fname in tqdm.tqdm(enumerate(lista_archivos)):
    with fits.open(fname) as hdul:
        img = hdul[0].data.astype(np.float32)

    bkg = sep.Background(img)
    img_sub = img - bkg.back()
    #plt.imshow(img_sub,origin='lower')
    #plt.show()
    # Perform aperture photometry at original object positions with dynamic radii
    flux, _,_ = sep.sum_circle(img_sub,objects['x'], objects['y'],r= r_apertures)

    fig, ax = plt.subplots()
    im = ax.imshow(img_sub, interpolation='nearest', cmap='gray', origin='lower')

    #for l in range(len(objects)):
    #    e = Ellipse(xy=(objects['x'][l], objects['y'][l]),
    #                width=6*objects['a'][l],
    #                height=6*objects['b'][l],
    #                angle=objects['theta'][l] * 180. / np.pi)
    #    e.set_facecolor('none')
    #    e.set_edgecolor('red')
    #    ax.add_artist(e)
#
    flux_matrix[i, :] = flux  # One row per image, one column per object

times=np.arange(len(flux_matrix[:,0]))*2 #seconds
for i in range(25):
    plt.plot(times,flux_matrix[:,i+100])#,label='object number:'+str(i+100))
    
plt.legend()
plt.ylabel("Total charge [electrons]")
plt.xlabel('Time [t]')
plt.show()

output_filename = _CURVITAS_DIR / "object_luminosity.fits"
hdu = fits.PrimaryHDU(flux_matrix)
hdu.writeto(output_filename, overwrite=True)
#plt.imshow(flux_matrix[])
#plt.show()