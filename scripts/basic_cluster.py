from pathlib import Path

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

def flatten(t):
    return [item for sublist in t for item in sublist]


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
fits_file = _PROJECT_ROOT / "FITS" / "skippercam_sim_v4.fits"

def basic_thresholding(threshold):
    n_grupos=0
    with fits.open(fits_file) as hdul:

        for i, hdu in enumerate(hdul):

            data = hdu.data

            if data is None:
                continue

            arr = np.asarray(data)
            #plt.imshow(data,origin='lower')
            #plt.show()
            #define a threshold
            #np.median(data)*1.3
            #plt.hist(flatten(data),histtype='step',bins=1000)
            #plt.show()
            thr_image=data>threshold
            #plt.imshow(thr_image,origin='lower')
            #plt.show()

            # 8-connectivity (all neighbors)
            structure = np.ones((3, 3), dtype=int)

            labels, nlabels = ndi.label(thr_image, structure=structure)
            #print(nlabels)

            n_grupos+=nlabels
            # Total charge in each group
            charges = ndi.sum(data, labels=labels, index=np.arange(1, nlabels + 1))
            #plt.hist(charges,histtype='step',bins=1000)
            #plt.show()
        # print(charges)
    print(n_grupos)
    return n_grupos

def multiple_thresholding(threshold_min,threshold_max,step):
    for threshold in np.arange(threshold_min,threshold_max,step):
        print(f"Threshold: {threshold}")
        basic_thresholding(threshold)
        for cluster in basic_thresholding(threshold):
            print(cluster)

    

def main():
    multiple_thresholding(130,200,10)

if __name__ == "__main__":
    main()