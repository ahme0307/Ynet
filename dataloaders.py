import sys, os
import numpy as np
from keras.preprocessing.image import transform_matrix_offset_center, apply_transform, Iterator,random_channel_shift, flip_axis
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import cv2
import random
import pdb
from skimage.io import imsave, imread
from skimage.transform import rotate
from skimage import transform
from skimage.transform import resize
from  params import * 
import math

def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)
def RandomLight(img):
    #lights = random.choice(["Rfilter","Rbright"])
    lights = random.choice(["Rfilter","Rbright","Rcontr", "RSat","RhueSat"])
    #print(lights)
    if lights=="Rfilter":
        alpha = 0.5 * random.uniform(0, 1)
        kernel = np.ones((3, 3), np.float32)/9 * 0.2
        colored = img[..., :3]
        colored = alpha * cv2.filter2D(colored, -1, kernel) + (1-alpha) * colored
        maxval = np.max(img[..., :3])
        dtype = img.dtype
        img[..., :3] = clip(colored, dtype, maxval)
    if lights=="Rbright":
        alpha = 1.0 + 0.1*random.uniform(-1, 1)
        maxval = np.max(img[..., :3])
        dtype = img.dtype
        img[..., :3] = clip(alpha * img[...,:3], dtype, maxval)
    if lights=="Rcontr":
        alpha = 1.0 + 0.1*random.uniform(-1, 1)
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        maxval = np.max(img[..., :3])
        dtype = img.dtype
        img[:, :, :3] = clip(alpha * img[:, :, :3] + gray, dtype, maxval)	
    if lights=="RSat":
        maxval = np.max(img[..., :3])
        dtype = img.dtype
        alpha = 1.0 + random.uniform(-0.1, 0.1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        img[..., :3] = alpha * img[..., :3] + (1.0 - alpha) * gray
        img[..., :3] = clip(img[..., :3], dtype, maxval)
    if lights=="RhueSat":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img)
        hue_shift = np.random.uniform(-25,25)
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(-25,25)
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(-25, 25)
        v = cv2.add(v, val_shift)
        img = cv2.merge((h, s, v))
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img

def perspectivedist(img,img_mask,flat_sum_mask, flag='all'):
    if flat_sum_mask>0 or flag=='all':           
            magnitude=8
           # pdb.set_trace()
            rw=img.shape[0]
            cl=img.shape[1]
          #  x = random.randrange(50, 200)
          #  nonzeromask=(img_mask>0).nonzero()
          #  nonzeroy = np.array(nonzeromask[0])
          #  nonzerox = np.array(nonzeromask[1])
          #  bbox = (( np.maximum(np.min(nonzerox)-x,0),  np.maximum(np.min(nonzeroy)-x,0)), (np.minimum(np.max(nonzerox)+x,cl),  np.minimum(np.max(nonzeroy)+x,rw)))
            #pdb.set_trace()
          #  img=img[bbox[0][1]:(bbox[1][1]),bbox[0][0]:(bbox[1][0])]
          #  img_mask=img_mask[bbox[0][1]:(bbox[1][1]),bbox[0][0]:(bbox[1][0])]
            skew = random.choice(["TILT", "TILT_LEFT_RIGHT", "TILT_TOP_BOTTOM", "CORNER"])
            w, h,_ = img.shape
            x1 = 0
            x2 = h
            y1 = 0
            y2 = w

            original_plane =  np.array([[(y1, x1), (y2, x1), (y2, x2), (y1, x2)]], dtype=np.float32)

            max_skew_amount = max(w, h)
            max_skew_amount = int(math.ceil(max_skew_amount *magnitude))
            skew_amount = random.randint(1, max_skew_amount)
            if skew == "TILT" or skew == "TILT_LEFT_RIGHT" or skew == "TILT_TOP_BOTTOM":
                if skew == "TILT":
                    skew_direction = random.randint(0, 3)
                elif skew == "TILT_LEFT_RIGHT":
                    skew_direction = random.randint(0, 1)
                elif skew == "TILT_TOP_BOTTOM":
                    skew_direction = random.randint(2, 3)

                if skew_direction == 0:
                    # Left Tilt
                    new_plane = np.array([(y1, x1 - skew_amount),  # Top Left
                                 (y2, x1),                # Top Right
                                 (y2, x2),                # Bottom Right
                                 (y1, x2 + skew_amount)], dtype=np.float32)  # Bottom Left
                elif skew_direction == 1:
                    # Right Tilt
                    new_plane = np.array([(y1, x1),                # Top Left
                                 (y2, x1 - skew_amount),  # Top Right
                                 (y2, x2 + skew_amount),  # Bottom Right
                                 (y1, x2)],dtype=np.float32)                # Bottom Left
                elif skew_direction == 2:
                    # Forward Tilt
                    new_plane = np.array([(y1 - skew_amount, x1),  # Top Left
                                 (y2 + skew_amount, x1),  # Top Right
                                 (y2, x2),                # Bottom Right
                                 (y1, x2)], dtype=np.float32)                # Bottom Left
                elif skew_direction == 3:
                    # Backward Tilt
                    new_plane = np.array([(y1, x1),                # Top Left
                                 (y2, x1),                # Top Right
                                 (y2 + skew_amount, x2),  # Bottom Right
                                 (y1 - skew_amount, x2)], dtype=np.float32)  # Bottom Left

            if skew == "CORNER":
                skew_direction = random.randint(0, 7)

                if skew_direction == 0:
                    # Skew possibility 0
                    new_plane = np.array([(y1 - skew_amount, x1), (y2, x1), (y2, x2), (y1, x2)], dtype=np.float32)
                elif skew_direction == 1:
                    # Skew possibility 1
                    new_plane = np.array([(y1, x1 - skew_amount), (y2, x1), (y2, x2), (y1, x2)], dtype=np.float32)
                elif skew_direction == 2:
                    # Skew possibility 2
                    new_plane = np.array([(y1, x1), (y2 + skew_amount, x1), (y2, x2), (y1, x2)],dtype=np.float32)
                elif skew_direction == 3:
                    # Skew possibility 3
                    new_plane = np.array([(y1, x1), (y2, x1 - skew_amount), (y2, x2), (y1, x2)], dtype=np.float32)
                elif skew_direction == 4:
                    # Skew possibility 4
                    new_plane = np.array([(y1, x1), (y2, x1), (y2 + skew_amount, x2), (y1, x2)], dtype=np.float32)
                elif skew_direction == 5:
                    # Skew possibility 5
                    new_plane = np.array([(y1, x1), (y2, x1), (y2, x2 + skew_amount), (y1, x2)], dtype=np.float32)
                elif skew_direction == 6:
                    # Skew possibility 6
                    new_plane = np.array([(y1, x1), (y2, x1), (y2, x2), (y1 - skew_amount, x2)],dtype=np.float32)
                elif skew_direction == 7:
                    # Skew possibility 7
                    new_plane =np.array([(y1, x1), (y2, x1), (y2, x2), (y1, x2 + skew_amount)], dtype=np.float32)
           # pdb.set_trace()
            perspective_matrix = cv2.getPerspectiveTransform(original_plane, new_plane)
            img = cv2.warpPerspective(img, perspective_matrix,
                                     (img.shape[1], img.shape[0]),
                                     flags = cv2.INTER_LINEAR)
            img_mask = cv2.warpPerspective(img_mask, perspective_matrix,
                                     (img.shape[1], img.shape[0]),
                                     flags = cv2.INTER_LINEAR)
    return img, img_mask
def add_gaussian_noise(X_imgs):

    #pdb.set_trace()
    row, col,_= X_imgs.shape
    #X_imgs=X_imgs/255
    X_imgs = X_imgs.astype(np.float32)
    # Gaussian distribution parameters
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    
    gaussian = np.random.random((row, col, 1)).astype(np.float32)
    gaussian = np.concatenate((gaussian, gaussian, gaussian), axis = 2)
    gaussian_img = cv2.addWeighted(X_imgs, 0.75, 0.25 * gaussian, 0.25, 0)
    gaussian_img = np.array(gaussian_img, dtype = np.uint8)
    return gaussian_img
def random_affine(img,img_mask):
    flat_sum_mask=sum(img_mask.flatten())
    (row,col)=img_mask.shape
    angle=shear_deg=0
    zoom=1
    center_shift   = np.array((1000, 1000)) / 2. - 0.5
    tform_center   = transform.SimilarityTransform(translation=-center_shift)
    tform_uncenter = transform.SimilarityTransform(translation=center_shift)
    big_img=np.zeros((1000,1000,3), dtype=np.uint8)
    big_mask=np.zeros((1000,1000), dtype=np.uint8)
    big_img[190:(190+row),144:(144+col)]=img
    big_mask[190:(190+row),144:(144+col)]=img_mask
    affine = random.choice(["rotate", "zoom", "shear"])
    if affine == "rotate":
        angle= random.uniform(-90, 90)
    if affine == "zoom":
        zoom = random.uniform(0.5, 1.5)
    if affine=="shear":
        shear_deg = random.uniform(-25, 25)    
   # pdb.set_trace()
    tform_aug = transform.AffineTransform(rotation = np.deg2rad(angle),
                                              scale =(1/zoom, 1/zoom),
                                              shear = np.deg2rad(shear_deg),
                                              translation = (0, 0))
    tform = tform_center + tform_aug + tform_uncenter
                   # pdb.set_trace()
    img_tr=transform.warp((big_img), tform)
    mask_tr=transform.warp((big_mask), tform)
                   # pdb.set_trace()
    masktemp =  cv2.cvtColor((img_tr*255).astype(np.uint8), cv2.COLOR_BGR2GRAY)>20
    img_tr=img_tr[np.ix_(masktemp.any(1),masktemp.any(0))]
    mask_tr = mask_tr[np.ix_(masktemp.any(1),masktemp.any(0))]
    return (img_tr*255).astype(np.uint8),(mask_tr*255).astype(np.uint8)    

class CustomNumpyArrayIterator(Iterator):

    def __init__(self, X, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 dim_ordering='th'):
        self.X = X
        self.y = y
        self.image_data_generator = image_data_generator
        self.dim_ordering = dim_ordering
        super(CustomNumpyArrayIterator, self).__init__(X.shape[0], batch_size, shuffle, seed)
    def _get_batches_of_transformed_samples(self, index_array):
       # pdb.set_trace()
        batch_x = np.zeros((len(index_array),img_rows,img_cols,3), dtype=np.float32)
        batch_y=np.zeros((len(index_array), img_rows,img_cols), dtype=np.float32)

        for i, j in enumerate(index_array):
            x = imread(self.X[j])
            y1 =imread(self.y[j])
            #print(j)
           # pdb.set_trace()

            _x, _y1 = self.image_data_generator.random_transform(x.astype(np.uint8), y1.astype(np.uint8))
            batch_x[i]=_x
            batch_y[i]=_y1
        batch_y=np.reshape(batch_y,(-1,img_rows,img_cols,1))
        return batch_x,batch_y

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)  
            #print(index_array)
        return self._get_batches_of_transformed_samples(index_array)
    

class CustomImageDataGenerator(object):
    def __init__(self, param):
        #netCLAHE=True, CROP=True, perspective=True,lighting=True,Flip=True,affine=True,randcrop=True
        self.CLAHE = param.CLAHE
        self.CROP = param.CROP
        self.perspective = param.perspective
        self.lighting = param.lighting
        self.Flip =param.Flip
        self.affine=param.affine
        self.randcrop=param.randcrop
        self.param=param
        
    
    def random_transform(self, img,img_mask):
        rw=img.shape[0]
        cl=img.shape[1]
        ch=np.shape(img.shape)[0]
        flag_crop=None
        if cl==1920:
            masktemp =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)>30
            img=img[np.ix_(masktemp.any(1),masktemp.any(0))]
            img_mask = img_mask[np.ix_(masktemp.any(1),masktemp.any(0))]
  
            img =cv2.resize(img, (img_rows,img_cols))
            img =np.squeeze(img)
               # img_mask = img_mask[:,300:]
            img_mask = cv2.resize(img_mask, (img_rows,img_cols))
            img_mask =np.squeeze(img_mask)    
        else:
            img =cv2.resize(img, (img_rows,img_cols))
            img_mask = cv2.resize(img_mask, (img_rows,img_cols))
            
        if np.shape(img_mask.shape)[0]==3:
            #pdb.set_trace()
            img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
        #img = img[:,:,:3]
        flat_sum_mask=sum(img_mask.flatten())  
        augCh = random.choice(["CROP","PER","ORIG", "FLIP","AFFINE","ORIG","randcrop","LIGHT"])
        if self.CLAHE and  augCh=="CLAHE":
            lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        if self.CROP and  augCh=="CROP":
            rw=img.shape[0]
            cl=img.shape[1]
            x = random.randrange(50, 200)
       
            if flat_sum_mask>0:
                #pdb.set_trace()
                bbox=[]
                nonzeromask=(img_mask>0).nonzero()
                nonzeroy = np.array(nonzeromask[0])
                nonzerox = np.array(nonzeromask[1])
                bbox = (( np.maximum(np.min(nonzerox)-x,0),  np.maximum(np.min(nonzeroy)-x,0)), (np.minimum(np.max(nonzerox)+x,cl),  np.minimum(np.max(nonzeroy)+x,rw)))
                #pdb.set_trace()
                img=img[bbox[0][1]:(bbox[1][1]),bbox[0][0]:(bbox[1][0])]
                img_mask=img_mask[bbox[0][1]:(bbox[1][1]),bbox[0][0]:(bbox[1][0])]
        if  self.perspective and augCh=="PER":
            #pdb.set_trace()
            img,img_mask=perspectivedist(img,img_mask,flat_sum_mask,'all')
            
        if self.affine and augCh=="AFFINE":
            #pdb.set_trace()
            img,img_mask=random_affine(img,img_mask)
           # pdb.set_trace()
        if self.lighting and augCh=="LIGHT":
             img = RandomLight(img)
        if self.Flip and augCh=="FLIP":
            flHV = random.choice(["H", "V"])
            if flHV=="H":
                #pdb.set_trace()
                img = cv2.flip( img, 0 )
                img_mask= cv2.flip( img_mask, 0)
                
            else:
                #pdb.set_trace()
                img = cv2.flip( img,1 )
                img_mask= cv2.flip( img_mask, 1)
        if self.randcrop and augCh=='randcrop':
            dx = dy = 112
            rx=random.randint(0, img_rows-dx-1)
            ry=random.randint(0, img_rows-dy-1)
            #pdb.set_trace()
            img = img[ry :ry +dy,  rx: rx+dx]
            img_mask=img_mask[ry :ry +dy,  rx: rx+dx]
        img= cv2.resize(img, (img_rows,img_cols))
        img_mask =  cv2.resize(img_mask, (img_rows,img_cols))

        img = img.astype('float32')
        img/=255.
        img_mask=img_mask.astype('float32')
        img_mask[img_mask>0] = 255.
        img_mask /= 255.  # scale masks to [0, 1]
       # pdb.set_trace()
        return np.array(img), np.array(img_mask)
    
    def flow(self, X, Y, batch_size, shuffle=True, seed=None):
        global img_rows
        global img_cols
        img_rows = self.param.img_rows
        img_cols = self.param.img_cols
        return CustomNumpyArrayIterator(
            X, Y, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed)

        

    
