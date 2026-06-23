from astropy.io import fits
import numpy as np
from scipy import ndimage
import scipy
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import skimage as sk
import skimage.io as io
import skimage.segmentation
import skimage.registration
from skimage.exposure import histogram
from skimage.morphology import disk
from scipy import ndimage as ndi
from glob import glob
from tqdm.auto import tqdm
from datetime import datetime
import os
from scipy.optimize import curve_fit
import ipywidgets as widgets
import math
import time
from skimage import io, color, morphology

def sum_intensity(region, intensities):
    return np.sum(np.round((intensities[region])))

def get_cluster_size_and_intensity(image,labels):
  rps=skimage.measure.regionprops(labels,intensity_image=image,cache=False, extra_properties=[sum_intensity])
  #areas=[r.area for r in rps]
  #print(rps.sum_intensity)
  energy=[r.sum_intensity for r in rps]
  dic_props={"energias":energy}
  return dic_props

def flatten(t):
    return [item for sublist in t for item in sublist]

def get_slice(arr, bounds):
  return arr[bounds[0]:bounds[1], bounds[2]:bounds[3]]

def select_event(image,labels,label):
  x_max=np.max(np.where(labels==label)[0])
  y_max=np.max(np.where(labels==label)[1])
  x_min=np.min(np.where(labels==label)[0])
  y_min=np.min(np.where(labels==label)[1])  
  return get_slice(image,[x_min-2,x_max+4,y_min-2,y_max+4])

def select_event_plot(image,labels,label):
  x_max=np.max(np.where(labels==label)[0])
  y_max=np.max(np.where(labels==label)[1])
  x_min=np.min(np.where(labels==label)[0])
  y_min=np.min(np.where(labels==label)[1])  
  return get_slice(image,[x_min-30,x_max+30,y_min-30,y_max+30])

def image_stdev(region, intensities):
    return np.std(intensities[region])

def image_intensity(region, intensities):
    return np.sum(intensities[region])

def intensity_max(region,intensities):
  return np.max(intensities[region])

def get_overscan(image,overscan):
  return image[:,overscan:]

def remove_baseline(image,overscan):
  ov=get_overscan(image,overscan)
  baselines=np.median(ov,axis=1)
  image_no_base=np.zeros(shape=np.shape(image))
  for i in range(0,len(baselines)):
    image_no_base[i]=image[i,:]-baselines[i]
  return image_no_base 

def any_pixel_above(matrix, value):  
  return np.any(matrix > value)

def generate_filtered(labels,lista_labels):
  _zero_img=np.zeros(shape=np.shape(labels),dtype=np.float32)
  for label in lista_labels:
    _x=np.where(labels==label)[0]
    _y=np.where(labels==label)[1]
    for i in range(len(_x)):
      _zero_img[_x[i],_y[i]]=1
  return _zero_img

def generate_filtered2(labels,lista_labels):
  _zero_img=np.zeros(shape=np.shape(labels),dtype=np.float32)
  for label in lista_labels:
    _x=np.where(labels==label+1)[0]
    _y=np.where(labels==label+1)[1]
    for i in range(len(_x)):
      _zero_img[_x[i],_y[i]]=1
  return _zero_img

##### Individual mask functions ###### ALL MASK SET TO 1 FOR ITERATION AND OPTIMIZATION THEN THEY ARE MULTIPLIED IN THE RESPECTIVE ANALYSIS CODE. :) 


def HEE_mask(image,thr_pix,thr_charge_ev):

  zero_img=np.zeros(shape=np.shape(image))
  label_im = ndimage.label(image>thr_pix,structure=[[1,1,1],[1,1,1],[1,1,1]])[0]
  energias=get_cluster_size_and_intensity(image,label_im)["energias"]
  hee_list=np.where(np.array(energias)>thr_charge_ev)[0].tolist()
  high_energy_evs=generate_filtered2(label_im,hee_list)

  zero_img[np.where(high_energy_evs>0)]=1

  return zero_img #masked_img 

def HEPixel_mask(image,thr_pix_energy):
  zero_img=np.zeros(shape=np.shape(image))
  label_im = image>thr_pix_energy
  zero_img[np.where(label_im>0)]=1
  return zero_img 

def crosstalk_mask(fits_name,thr_crosstalk):

  hdu_list=fits.open(fits_name)
  
  imgs_masks=[]

  for hdu in range(len(hdu_list)):
    
    image_data=hdu_list[hdu].data
    hee_mask=image_data>thr_crosstalk

    imgs_masks.append(hee_mask)
  
  hdu_list.close()

  crosst_=np.zeros(shape=np.shape(imgs_masks[0]))
  zero_img=np.zeros(shape=np.shape(imgs_masks[0]))

  for hdu in range(2):
    crosst_+=np.array(imgs_masks)[hdu]

  zero_img[np.where(crosst_>0)]=1

  return zero_img

def halo_mask(image,thr_charge_ev,iterat,serialz):
    zero_img=np.zeros(shape=np.shape(image))
    
    image[serialz==1]=0
    
    selem = disk(iterat)
    
    dilated_mask = ndi.morphology.binary_dilation(image, selem)

    halo=dilated_mask
    
    zero_img[np.where(halo>0)]=1

    return zero_img

def bleed_mask(image,thr_charge_ev,bleed_iter,direction,serialz):
    
    zero_img=np.zeros(shape=np.shape(image))


    if direction =="y":
        image[serialz==1]=0
        dilated_mask=ndi.morphology.binary_dilation(image,structure=[[0,0,0],[0,1,0],[0,1,0]],iterations=bleed_iter)#structure=[[1,1,1],[1,1,1],[1,1,1]]
    elif direction =="x":
        dilated_mask=ndi.morphology.binary_dilation(image,structure=[[0,0,0],[0,1,1],[0,0,0]],iterations=bleed_iter)    
    else:
        dilated_mask=np.zeros(shape=image)
        print("Specify bleed direction")

    zero_img[np.where(dilated_mask>0)]=1

    return zero_img 
   


#def bleed_maskx(image,thr_charge_ev,bleed_iter,hotcol_mask):
#    new_im=(image*(hotcol_mask==0))
#    zero_img=np.zeros(shape=np.shape(image))
#    dilated_mask=ndi.morphology.binary_dilation(new_im>thr_charge_ev,structure=[[0,0,0],[0,1,1],[0,0,0]],iterations=bleed_iter)    
#    zero_img[np.where(dilated_mask>0)]=1
#    return zero_img 
#
#def bleed_masky(image,thr_charge_ev,bleed_iter,serial_mask):
#    new_im=(image*(serial_mask==0))
#    zero_img=np.zeros(shape=np.shape(image))
#    dilated_mask=ndi.morphology.binary_dilation(new_im>thr_charge_ev,structure=[[0,0,0],[0,1,0],[0,1,0]],iterations=bleed_iter)#structure=[[1,1,1],[1,1,1],[1,1,1]]
#    zero_img[np.where(dilated_mask>0)]=1
#
#    return zero_img 
   

def pres_overs_border_mask(image,prescan,overscan,borders):
  zero_img=np.zeros(shape=np.shape(image))
  other_bord=np.shape(image)[0]-borders
  
  zero_img[:,:prescan]  = 1
  zero_img[:,overscan:] =  1
  zero_img[:borders,:]  = 1
  zero_img[other_bord:,:] = 1

  return zero_img

def check_file_in_folder(folder_path, file_name):
  file_path = os.path.join(folder_path, file_name)
  return os.path.exists(file_path)





def filtrar_labels_planoides(image,pix_thr):
    image[:,:11]=0
    
    lista_filtrin=[]
    binary=image>pix_thr 
    _label_im=ndimage.label(morphology.remove_small_objects(binary, 2),structure=[[1,1,1],[1,1,1],[1,1,1]])[0]
    zero_img=np.zeros(shape=np.shape(image),dtype=np.float32)

    #plt.imshow(_label_im>0)
    #plt.show()

#    coordin=skimage.measure.regionprops(label_im, intensity_image=None)
#    print(coordin[0].coords[0][1])
#    for m in tqdm(np.unique(label_im)[1:]):
#        for y_pix in coordin[m].coords[0]:
#            print(len(y_pix))

    for m in np.unique(_label_im)[1:]:
        _track=_label_im==m
        pixeles_x=np.unique(np.where(_track)[1])
        pixeles_y=np.unique(np.where(_track)[0])

        if (len(pixeles_y)>3):
            zero_img[np.where(_track)[0],np.where(_track)[1]]=1
            lista_filtrin.append(m)


 #   print(lista_filtrin)
#    plt.imshow(zero_img)
#    plt.show()

#    plt.imshow(label_im>0)
#    plt.show()

    #plt.imshow(generate_filtered(_label_im,lista_filtrin),origin='lower')
    #plt.show()

    dilated_mask=generate_filtered(_label_im,lista_filtrin)

    return dilated_mask

def sum_pixels_in_windows(array, window_size):
    height, width = array.shape
    
    new_width = width // window_size
    
    new_array = np.zeros((height, new_width), dtype=np.float32)
    
    for row in range(height):
        for col in range(new_width):
            start_index = col * window_size
            end_index = start_index + window_size
            new_array[row, col] = array[row, start_index:end_index].sum()
    return new_array

def add_dc_to_img(img,Dc_events,num_pixels,overscan_size):
    imagen_dc = np.zeros(num_pixels)  
    uni_int_1=np.random.randint(0,num_pixels[0],size=(int(round(Dc_events))))
    uni_int_2=np.random.randint(prescan,num_pixels[1]-overscan_size,size=(int(round(Dc_events))))
    for i in range(int(round(Dc_events))):
        imagen_dc[uni_int_1[i],uni_int_2[i]]+=1
    return img+imagen_dc

def add_read_to_img(img,readout_nos,num_pixels):
    read_out_img=np.random.normal(0,scale=readout_nos,size=(num_pixels))
    return img+read_out_img


def simulate_img(num_pix,mu_rate):
    _zeros=np.zeros(num_pix)
    
    _imagen_dc=add_dc_to_img(_zeros,mu_rate,num_pix,overscan_size)

    return _imagen_dc

def select_event(image,labels,label):
  x_max=np.max(np.where(labels==label)[0])
  y_max=np.max(np.where(labels==label)[1])
  x_min=np.min(np.where(labels==label)[0])
  y_min=np.min(np.where(labels==label)[1])  
  return get_slice(image,[x_min-2,x_max+4,y_min-10,y_max+10])
  

overscan_size=150
prescan=7
dist_tol=30

shifts=np.arange(0,50,10)

def get_slice_rows(image,labels,label):
    y_max=np.max(np.where(labels==label)[1])
    y_min=np.min(np.where(labels==label)[1])  

    return  image[y_min:y_max,:]

def srh_mask2(image,pix_thr,hotcols,tgate):

  image[:,:11]=0

  zero_img=np.zeros(shape=np.shape(image),dtype=np.int8)

  #plt.imshow(hotcols)
  #plt.show()

  pre_img=(image)*(filtrar_labels_planoides(image,pix_thr)==0)*(hotcols==0)*(tgate==0)         # # # # # # # # # # # # # # # # # (image<pix_thr*6)  # # # # # # # # # # # # # # # # # 

  binary=(pix_thr<pre_img) #& (pre_img<1.63)
  #plt.imshow(binary)
  #plt.show()
  label_full=ndimage.label(binary,structure=[[1,1,1],[1,1,1],[1,1,1]])[0]

  filter_small=morphology.remove_small_objects(binary,2)
  

  label_im=ndimage.label(filter_small,structure=[[1,1,1],[1,1,1],[1,1,1]])[0]

  small_obj=binary*1.0-filter_small*1.0

  centrus=skimage.measure.regionprops(label_im)

  ev_imctr=np.zeros(shape=np.shape(image),dtype=np.int8)
  plt.imshow(small_obj,origin='lower')
  plt.show()
  
  for m in np.unique(label_im)[1:]:
    tupla=centrus[m-1].centroid
    ev_imctr[int(np.round(tupla[0])),int(np.round(tupla[1]))]=1
  
  for m in np.unique(label_im)[1:]:
    
    track=(label_im==m)
    pixeles_x=np.unique(np.where(track)[1])
    pixeles_y=np.unique(np.where(track)[0])
    carga_y=track[pixeles_y,:]
    event_tag=np.sum(carga_y,axis=1)


    if (len(pixeles_x)>7):
        zero_img[pixeles_y,:]=1
    
    if (len(pixeles_y)==1) & (len(pixeles_x)>4):
        zero_img[pixeles_y,:]=1

    if (len(pixeles_y)==1) or ((len(pixeles_y)==2) and ((event_tag[0]==1 and event_tag[1]>1) or (event_tag[1]==1 and event_tag[0]>1))) or (len(pixeles_y)==3 and ((event_tag[0]<=1) and (event_tag[1]>2) and (event_tag[2]<=1))):
      if (len(pixeles_y)==1):
        relevant_select=pixeles_y[0]
      if len(pixeles_y)>1:
        relevant_select=pixeles_y[np.where(event_tag==max(event_tag))][0]

      p1=min(pixeles_x)
      p2=max(pixeles_x)
      try:
          dist1=min(abs(np.where(binary[relevant_select,:p1])[0]-p1)-1)
          pos_1= p1-dist1-1
          label_indic=label_full[relevant_select,pos_1]
          pixeles_y_cand=np.unique(np.where(label_indic==label_full)[0])
          if dist1<dist_tol & len(pixeles_y_cand)==1:
              for g in pixeles_y:
                  zero_img[g,:]=1
          else:
             try:
              dist1_new=dist1+min(abs(np.where(binary[pixeles_y_cand[0],:pos_1])[0]-pos_1)-1)
             except:
              dist1_new=dist_tol+1
      except:
          dist1=dist_tol+1
      try:
          dist2=min(np.where(binary[relevant_select,p2+1:])[0])
    
          pos_2= p2+dist2+1
          
          label_indic=label_full[relevant_select,pos_2]
          pixeles_y_cand=np.unique(np.where(label_indic==label_full)[0])
    
          if dist2<dist_tol & len(pixeles_y_cand)==1:
              for g in pixeles_y:
                  zero_img[g,:]=1
          else:
             try:
              dist2_new=dist1+min(abs(np.where(binary[pixeles_y_cand[0],:pos_2])[0]+pos_2)-1)
             except:
              dist2_new=dist_tol+1
      except:
          dist2=dist_tol+1 
      if  (dist1<dist_tol) or (dist2<dist_tol): 
          for g in pixeles_y:
              zero_img[g,:]=1
    
  for shift in shifts:

    binned_shifted=sum_pixels_in_windows((zero_img[:,shift:]==0)*(ev_imctr[:,shift:]+small_obj[:,shift:]),60)

    if any_pixel_above(binned_shifted,3):
      for k in np.where(binned_shifted>3)[0]:
          zero_img[k,:]=1

  small_remaining=(zero_img==0)*small_obj*pre_img
  if np.sum(small_remaining>pix_thr*3.5)>0:
      
    label_im_1pixhee=ndimage.label(small_remaining>pix_thr*3.5,structure=[[1,1,1],[1,1,1],[1,1,1]])[0]

    for h in np.unique(label_im_1pixhee)[1:]:
      track=(label_im_1pixhee==h)
      pixeles_x=np.unique(np.where(track)[1])
      pixeles_y=np.unique(np.where(track)[0])

      try:
          dist1=min(abs(np.where(binary[pixeles_y[0],:pixeles_x[0]])[0]-pixeles_x[0])-1)
      except:
          dist1=dist_tol+1
      try:
          dist2=min(np.where(binary[pixeles_y[0],pixeles_x[0]+1:])[0])
      except:
          dist2=dist_tol+1

      if  (dist1<dist_tol) or (dist2<dist_tol): 
          zero_img[pixeles_y,:]=1
#  plt.imshow(zero_img*1.0+(image>0.7)*1.0)
#  plt.show()
  return zero_img

def get_cluster_E_and_x(image,labels):
  energy=[]
  L=[]
  for m in np.unique(labels)[1:]:
    track=(labels==m)

    pixeles_y=np.unique(np.where(track)[0])
    pixeles_x=np.unique(np.where(track)[1])
    
#  print(np.sum(image*track))
    if len(pixeles_y==1):
      L.append(len(pixeles_x))
      energy.append(np.sum(image*track))
  
  #print(L)
  #print(energy)
  
  return energy,L

def srh_mask2_L_E(image,pix_thr,hotcols,tgate):

  image[:,:11]=0

  zero_img=np.zeros(shape=np.shape(image),dtype=np.int8)

  #plt.imshow(hotcols)
  #plt.show()

  pre_img=(image)*(filtrar_labels_planoides(image,pix_thr)==0)*(hotcols==0)*(tgate==0)         # # # # # # # # # # # # # # # # # (image<pix_thr*6)  # # # # # # # # # # # # # # # # # 

  binary=(pix_thr<pre_img) #& (pre_img<1.63)
  #plt.imshow(binary)
  #plt.show()
  label_full=ndimage.label(binary,structure=[[1,1,1],[1,1,1],[1,1,1]])[0]

  filter_small=morphology.remove_small_objects(binary,2)
  

  label_im=ndimage.label(filter_small,structure=[[1,1,1],[1,1,1],[1,1,1]])[0]

  small_obj=binary*1.0-filter_small*1.0

  centrus=skimage.measure.regionprops(label_im)

  ev_imctr=np.zeros(shape=np.shape(image),dtype=np.int8)
  
  
  for m in np.unique(label_im)[1:]:
    tupla=centrus[m-1].centroid
    ev_imctr[int(np.round(tupla[0])),int(np.round(tupla[1]))]=1
  
  for m in np.unique(label_im)[1:]:
    
    track=(label_im==m)
    pixeles_x=np.unique(np.where(track)[1])
    pixeles_y=np.unique(np.where(track)[0])
    carga_y=track[pixeles_y,:]
    event_tag=np.sum(carga_y,axis=1)


    if (len(pixeles_x)>7):
        zero_img[pixeles_y,:]=1
    
    if (len(pixeles_y)==1) & (len(pixeles_x)>4):
        zero_img[pixeles_y,:]=1

    if (len(pixeles_y)==1) or ((len(pixeles_y)==2) and ((event_tag[0]==1 and event_tag[1]>1) or (event_tag[1]==1 and event_tag[0]>1))) or (len(pixeles_y)==3 and ((event_tag[0]<=1) and (event_tag[1]>2) and (event_tag[2]<=1))):
      if (len(pixeles_y)==1):
        relevant_select=pixeles_y[0]
      if len(pixeles_y)>1:
        relevant_select=pixeles_y[np.where(event_tag==max(event_tag))][0]

      p1=min(pixeles_x)
      p2=max(pixeles_x)
      try:
          dist1=min(abs(np.where(binary[relevant_select,:p1])[0]-p1)-1)
          pos_1= p1-dist1-1
          label_indic=label_full[relevant_select,pos_1]
          pixeles_y_cand=np.unique(np.where(label_indic==label_full)[0])
          if dist1<dist_tol & len(pixeles_y_cand)==1:
              for g in pixeles_y:
                  zero_img[g,:]=1
          else:
             try:
              dist1_new=dist1+min(abs(np.where(binary[pixeles_y_cand[0],:pos_1])[0]-pos_1)-1)
             except:
              dist1_new=dist_tol+1
      except:
          dist1=dist_tol+1
      try:
          dist2=min(np.where(binary[relevant_select,p2+1:])[0])
    
          pos_2= p2+dist2+1
          
          label_indic=label_full[relevant_select,pos_2]
          pixeles_y_cand=np.unique(np.where(label_indic==label_full)[0])
    
          if dist2<dist_tol & len(pixeles_y_cand)==1:
              for g in pixeles_y:
                  zero_img[g,:]=1
          else:
             try:
              dist2_new=dist1+min(abs(np.where(binary[pixeles_y_cand[0],:pos_2])[0]+pos_2)-1)
             except:
              dist2_new=dist_tol+1
      except:
          dist2=dist_tol+1 
      if  (dist1<dist_tol) or (dist2<dist_tol): 
          for g in pixeles_y:
              zero_img[g,:]=1
    
  for shift in shifts:

    binned_shifted=sum_pixels_in_windows((zero_img[:,shift:]==0)*(ev_imctr[:,shift:]+small_obj[:,shift:]),60)

    if any_pixel_above(binned_shifted,3):
      for k in np.where(binned_shifted>3)[0]:
          zero_img[k,:]=1

  small_remaining=(zero_img==0)*small_obj*pre_img
  
  
  if np.sum(small_remaining>pix_thr*3.5)>0:
      
    label_im_1pixhee=ndimage.label(small_remaining>pix_thr*3.5,structure=[[1,1,1],[1,1,1],[1,1,1]])[0]

    for h in np.unique(label_im_1pixhee)[1:]:
      track=(label_im_1pixhee==h)
      pixeles_x=np.unique(np.where(track)[1])
      pixeles_y=np.unique(np.where(track)[0])

      try:
          dist1=min(abs(np.where(binary[pixeles_y[0],:pixeles_x[0]])[0]-pixeles_x[0])-1)
      except:
          dist1=dist_tol+1
      try:
          dist2=min(np.where(binary[pixeles_y[0],pixeles_x[0]+1:])[0])
      except:
          dist2=dist_tol+1

      if  (dist1<dist_tol) or (dist2<dist_tol): 
          zero_img[pixeles_y,:]=1



  label_finail_hist=ndimage.label(zero_img*1.0*(image>0.7)*1.0,structure=[[1,1,1],[1,1,1],[1,1,1]])[0]
  Ener,Longi=get_cluster_E_and_x(image,label_finail_hist)
  #plt.imshow(label_finail_hist)
  #plt.show()
  
  energia_ser=[]
  length_ser=[]

  return Ener,Longi

def tgate_bleedmask(image,thr_pix,tgatetrap_loc,HDU):
  zero_img=np.zeros(shape=np.shape(image),dtype=np.int8)
  if HDU==0:
    binary=image>(thr_pix*5)## THRESHOLD """"" HIGH ENERGY """
    if len(np.where(binary[:,tgatetrap_loc]==True)[0])>0:
      for fila in np.where(binary[:,tgatetrap_loc]==True):
        zero_img[fila,tgatetrap_loc:]=1
        try:
          zero_img[fila-1,tgatetrap_loc:]=1
          zero_img[fila+1,tgatetrap_loc:]=1
        except:
            pass

  return zero_img 


############################## Func to select threhold ID and EXT: ##############################

def get_thr(ID,EXT,Run):
  if Run!='Run_celeste':

    if (ID>=9) & (ID<=382)    & (EXT==0):
      pix_thr=0.76

    if (ID>=383) & (ID<=551)    & (EXT==0):
      pix_thr=0.77

    if (ID>=552) & (ID<=629)  & (EXT==0):
      pix_thr=0.78

    if (ID>=630) & (ID<=671)  & (EXT==0):
      pix_thr=0.79

    if (ID>=672) & (ID<=703)  & (EXT==0):
      pix_thr=0.80

    if (ID>=704) & (ID<=786)  & (EXT==0):
      pix_thr=0.81  

    if (ID>=787) & (ID<=815)  & (EXT==0):
      pix_thr=0.8

    if (ID>=816) & (ID<=856)  & (EXT==0):
      pix_thr=0.79

    if (ID>=857) & (ID<=929)  & (EXT==0):
      pix_thr=0.78

    if (ID>=930) & (ID<=1299)  & (EXT==0):
      pix_thr=0.77

    if (ID>=1300) & (ID<=1321)  & (EXT==0):
      pix_thr=0.78

    if (ID>=1322) & (ID<=1354) & (EXT==0):
      pix_thr=0.77

    if (ID>=1355) & (ID<=1513) & (EXT==0):
      pix_thr=0.67

##  ############################## LA OTRA EXT ################################

    if (ID>=9) & (ID<=437)  & (EXT==1):
      pix_thr=0.79

    if (ID>=438) & (ID<=564)  & (EXT==1):
      pix_thr=0.8

    if (ID>=565) & (ID<=626)  & (EXT==1):
      pix_thr=0.81

    if (ID>=627) & (ID<=643)  & (EXT==1):
      pix_thr=0.82

    if (ID>=644) & (ID<=668)  & (EXT==1):
      pix_thr=0.83

    if (ID>=669) & (ID<=685)  & (EXT==1):
      pix_thr=0.84

    if (ID>=686) & (ID<=703)  & (EXT==1):
      pix_thr=0.85

    if (ID>=704) & (ID<=717)  & (EXT==1):
      pix_thr=0.86

    if (ID>=718) & (ID<=781)  & (EXT==1):
      pix_thr=0.87

    if (ID>=782) & (ID<=802)  & (EXT==1):
      pix_thr=0.86

    if (ID>=803) & (ID<=825)  & (EXT==1):
      pix_thr=0.85

    if (ID>=826) & (ID<=853)  & (EXT==1):
      pix_thr=0.84
#######################
    if (ID>=854) & (ID<=1251)  & (EXT==1):
      pix_thr=0.83

#    if (ID>=993) & (ID<=1252)  & (EXT==1):
#      pix_thr=0.83

    if (ID>=1252)  & (ID<=1354) & (EXT==1):
      pix_thr=0.84

    if (ID>=1355)  & (ID<=1386) & (EXT==1):
      pix_thr=0.70

    if (ID>=1387) & (ID<=1469) & (EXT==1):
      pix_thr=0.71

    if (ID>=1470) & (ID<=1513) & (EXT==1):
      pix_thr=0.72

  else:  
    #print('Procesando Run celeste:')
    if (ID>=19) & (ID<=78) & (EXT==0):
      pix_thr=0.67
    if (ID>=78) & (ID<=217) & (EXT==0):
      pix_thr=0.68
    if (ID>=218)  & (EXT==0):
      pix_thr=0.67
#### la otra ext
    if (ID>=19) & (ID<=27) & (EXT==1):
      pix_thr=0.71
    if (ID>=28) & (ID<=37) & (EXT==1):
      pix_thr=0.72
    if (ID>=38) & (ID<=59) & (EXT==1):
      pix_thr=0.73
    if (ID>=60) & (ID<=106) & (EXT==1):
      pix_thr=0.75

    if (ID>=107) & (ID<=207) & (EXT==1):
      pix_thr=0.76

    if (ID>=208) & (ID<=257) & (EXT==1):
      pix_thr=0.75

    if (ID>=258) & (EXT==1):
      pix_thr=0.74
    #print(pix_thr)
    #print(ID)
  return pix_thr



######################## Esta funcion es solo para moskita ########################

##FLAGS:
##HEE=2
##CROSSTALK=4
##TGATE=8
##SERIAL=16
##BORDER=32
##HALO=64
##BLEEDX=128
##BLEEDY=256
##HOTCOL1=512
##

def generate_mask_LEB(fits_file,ID,hee_thr,cross_thr,max_iterations,bleed_x,bleed_y,prescan,overscan,borders,folder_with_fits,hotcols_mask,tgatetrap_loc,Run,save=True):
  filename=fits_file.split(".fits")[0].split(folder_with_fits+"/")[1]

  
  img_list=[]  
  hdu_list=fits.open(fits_file)

  for k in [0,1]:
        img_list.append(hdu_list[k].data)
  hdu_list.close()
  
  new_hdul = fits.HDUList()

  cross_masked=crosstalk_mask(fits_file,cross_thr)
  
  for i in range(len(img_list)):
    if Run=='Run_violeta':
      if i==0:
        pix_thr=0.67
      else:
        pix_thr=0.72

    else:
      pix_thr=get_thr(ID,i,Run)
    hee_masked_2=HEPixel_mask(img_list[i],hee_thr)
    tgate_masked = tgate_bleedmask(img_list[i],pix_thr,tgatetrap_loc,i)
    serial_mask=srh_mask2(img_list[i],pix_thr,hotcols_mask[i],tgate_masked)
    border_masked=pres_overs_border_mask(img_list[i],prescan,overscan,borders)  
    halo_masked=halo_mask(hee_masked_2,hee_thr,max_iterations,serial_mask)
    bleed_masked_x=bleed_mask(hee_masked_2,hee_thr,bleed_x,'x',serial_mask)
    bleed_masked_y=bleed_mask(hee_masked_2,hee_thr,bleed_y,'y',serial_mask)
    hotcols_both=hotcols_mask[i]
    HEE_clust=HEE_mask(img_list[i],pix_thr,hee_thr)
    
    mask_image=cross_masked*8+tgate_masked*1024+serial_mask*16+border_masked*32+hee_masked_2*2+halo_masked*64+bleed_masked_x*128+bleed_masked_y*256+hotcols_both*512+HEE_clust*4

    new_hdul.append(fits.ImageHDU(np.int16(mask_image)))

  if save:
    try:
      new_hdul.writeto("mask_"+filename+".fits",overwrite=True)
    except:
      pass
    return None 
  return new_hdul



def generate_mask_SER(fits_file,ID,hee_thr,cross_thr,max_iterations,bleed_x,bleed_y,prescan,overscan,borders,folder_with_fits,hotcols_mask,tgatetrap_loc,Run,save=True):
  filename=fits_file.split(".fits")[0].split(folder_with_fits+"/")[1]

  
  img_list=[]  
  hdu_list=fits.open(fits_file)

  for k in [0,1]:
        img_list.append(hdu_list[k].data)
  hdu_list.close()
  
  new_hdul = fits.HDUList()

  cross_masked=crosstalk_mask(fits_file,cross_thr)
  
  hotcols_both=((hotcols_mask[0]+hotcols_mask[1])>0)
  
  for i in range(len(img_list)):
    if Run=='Run_violeta':
      if i==0:
        pix_thr=0.67
      else:
        pix_thr=0.72
    else:
      pix_thr=get_thr(ID,i,Run)


    hee_masked_2=HEPixel_mask(img_list[i],hee_thr)
    tgate_masked = tgate_bleedmask(img_list[i],pix_thr,tgatetrap_loc,i)
    serial_mask=srh_mask2(img_list[i],pix_thr,hotcols_mask[i],tgate_masked)
    #border_masked=pres_overs_border_mask(img_list[i],prescan,overscan,borders)  
    #halo_masked=halo_mask(hee_masked_2,hee_thr,max_iterations,serial_mask)
    #bleed_masked_x=bleed_mask(hee_masked_2,hee_thr,bleed_x,'x',serial_mask)
    #bleed_masked_y=bleed_mask(hee_masked_2,hee_thr,bleed_y,'y',serial_mask)

    mask_image=serial_mask*16

    new_hdul.append(fits.ImageHDU(np.int16(mask_image)))

  if save:
    try:
      new_hdul.writeto("ser_mask_"+filename+".fits",overwrite=True)
    except:
      pass
    return None 
  return new_hdul




def get_only_ser(fits_file,ID,hee_thr,cross_thr,max_iterations,bleed_x,bleed_y,prescan,overscan,borders,folder_with_fits,hotcols_mask,tgatetrap_loc,save=True):
  filename=fits_file.split(".fits")[0].split(folder_with_fits+"/")[1]

  
  img_list=[]  
  hdu_list=fits.open(fits_file)

  for k in [0,1]:
        img_list.append(hdu_list[k].data)
  hdu_list.close()
  
  new_hdul = fits.HDUList()

  cross_masked=crosstalk_mask(fits_file,cross_thr)
  
  hotcols_both=((hotcols_mask[0]+hotcols_mask[1])>0)
  
  for i in range(len(img_list)):
    pix_thr=get_thr(ID,i,Run)
    hee_masked_2=HEPixel_mask(img_list[i],hee_thr)
    tgate_masked = tgate_bleedmask(img_list[i],pix_thr,tgatetrap_loc,i)
    SR_E,SR_L=srh_mask2_L_E(img_list[i],pix_thr,hotcols_mask[i],tgate_masked)
    #border_masked=pres_overs_border_mask(img_list[i],prescan,overscan,borders)  
    #halo_masked=halo_mask(hee_masked_2,hee_thr,max_iterations,serial_mask)
    #bleed_masked_x=bleed_mask(hee_masked_2,hee_thr,bleed_x,'x',serial_mask)
    #bleed_masked_y=bleed_mask(hee_masked_2,hee_thr,bleed_y,'y',serial_mask)

   # mask_image=serial_mask*16
##
  #  new_hdul.append(fits.ImageHDU(np.int16(mask_image)))
###
  return SR_E,SR_L
