from __future__ import print_function
import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import pdb
from skimage.io import imsave, imread
import cv2
#import pylab
import imageio
#import matplotlib.pyplot as plt
import params 
from  get_resUnet import *
#from  get_model import *
from natsort import natsorted
from os.path import splitext
#from keras.utils import plot_model
#import pylab as plt
from  gen_data import load_image,random_batch,test_batch,load_images,plot_images,load_test_image
from params import *
from skimage.io import imsave, imread
from skimage.io import imsave
#set the source directory here
from skimage.draw import ellipse
from skimage.measure import label, regionprops
import csv

def findcenters2(img):
    #img=(img>0.2).astype(np.float)
    match_pxls = np.where(img > 0.9)
            
    B = np.zeros_like(img).astype(np.float)
    B[match_pxls] = 1 
    img=B
    img=(img).astype(np.uint8)
    #img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    temp=img
    label_img = label(img)
    regions = regionprops(label_img)
    coorinates=[]
    detection =0
   # fig, ax = plt.subplots()
   # ax.imshow(img, cmap=plt.cm.gray)
    temp_area=0

    for props in regions:
        y0, x0 = props.centroid
        if props.area >=temp_area and temp_area==0:
            coorinates.append((y0,x0))
            temp_area=props.area
            detection=1
        elif props.area >=temp_area:
            coorinates[0]=(y0,x0)
            temp_area=props.area
    return coorinates,detection

def predict_for_image(i,files,data_path,pred_dir,framestat,model,writer,save_result=True):
    imtest=os.path.join(data_path,files[i])
    imtest,infos=load_test_image([imtest])
    
    pred = model.predict(imtest, batch_size=1,verbose=0)
    imtest= (imtest*255.).astype(np.uint8)
    imtest  = np.array(np.squeeze(imtest),dtype= np.uint8)
    pred =np.squeeze(pred)
    base=os.path.splitext(files[i])[0]
   # y_pred_f = (pred>0.5).flatten()
   # if sum(y_pred_f)==0:
   #     detection=0
   # else:
   #     detection=1   
    pred =cv2.resize(pred,( infos[0][1],infos[0][0]),interpolation=cv2.INTER_CUBIC)
    temp_coord=[]
    temp_coord,detection=findcenters2(pred)

    imtest =cv2.resize(imtest,( infos[0][1],infos[0][0]),interpolation=cv2.INTER_CUBIC)
    path=os.path.join(pred_dir,base+'_mask'+'.png')
    pred=np.clip(pred, 0, 1).astype(np.float)
    im_pred = np.array(255*pred,dtype=np.uint8)
    rgb_mask_pred = cv2.cvtColor(im_pred,cv2.COLOR_GRAY2RGB)
    rgb_mask_pred[:,:,1:2] = 0*rgb_mask_pred[:,:,1:2]
    heatmap = im_pred
    imtest= (np.squeeze(imtest)*255).astype(np.uint8)
    #

    draw_img = cv2.addWeighted(rgb_mask_pred,0.2,imtest,1,0)
    for props in temp_coord:
        cv2.circle(draw_img, (int(props[1]),int(props[0])), 7, (0, 0, 1), -1)
        framestat.append(i+1)
        framestat.append(detection)
        framestat.append(props[0] )
        framestat.append(props[1] )
        pred_conf=pred[pred>0.2]
        framestat.append(max(0,pred_conf[np.nonzero(pred_conf)].mean()))
    if len(temp_coord)==0:
        framestat.append(i+1)
        framestat.append( 0 )
        framestat.append(0)
        framestat.append( 0 )
        pred_conf=pred[pred>0.05]
        framestat.append(1-max(0,pred_conf[np.nonzero(pred_conf)].mean()) )               
    #pdb.set_trace()
  
    if save_result:
        #pdb.set_trace()
        image = cv2.cvtColor((imtest*255).astype('uint8'), cv2.COLOR_RGB2BGR);
        output = np.zeros((infos[0][0],infos[0][1]*3, 3), dtype='uint8')
        output[0:infos[0][0],0:infos[0][1],:]=cv2.cvtColor((imtest*255).astype('uint8'), cv2.COLOR_RGB2BGR);
        output[0:infos[0][0],(infos[0][1]):(2*infos[0][1])]=cv2.cvtColor((imtest*np.dstack([pred]*3)*255).astype('uint8'), cv2.COLOR_RGB2BGR); 
        output[0:infos[0][0],2*(infos[0][1]):3*(infos[0][1])]=cv2.cvtColor((draw_img*255).astype('uint8'), cv2.COLOR_RGB2BGR); 
        #imsave(path,draw_img*255 )
       # pdb.set_trace()
        writer.write(output)
    #framestat.append((base,detection,temp_coord))
    return framestat
    


if __name__ == "__main__":
    params.init() 
    params.settings['rotzoom']=0
    params.settings['CLAHE']=0
    params.settings['OnlyPost']=1
    params.settings['crop']=0
    params.settings['training']=0
    params.settings['lighting']=0
    params.settings['perspective']=0
   # model =Y10_net()
   # filename='Y10_net.hdf5'
    #logdirs='%s%s'%(splitext(filename)[0],'logs')
    model =YnetResNet(include_top=False, weights='imagenet')
    filename='YnetResNetFinal.hdf5'
    #model =YnetResNet(include_top=False, weights='imagenet')
    #filename='YnetResNetFinal.hdf5'
    model.load_weights(filename, by_name=True)
    dete_sub ='./detection/'
    local_sub='./localize/'
    for folder in range(1,19):
        folder_name=str(folder)
 
        data_path = os.path.join(    '/media/a252/540/ChallengeTest/TestPolyDet',folder_name)
        pred_dir=os.path.join(  '/media/a252/540/ChallengeTest/Pred',folder_name)
        files = [f for f in os.listdir(data_path) if f[-3:] == 'png']
        files=natsorted(files, key=lambda y: y.lower())
        framestat=[]
        # initialize the FourCC, video writer, dimensions of the frame, and
        # zeros array
        imtest=os.path.join(data_path,files[0])
        imtest,infos=load_test_image([imtest])
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        
        writer = cv2.VideoWriter('Video'+folder_name+'.avi', fourcc, 20.0,(infos[0][1]*3,infos[0][0]))
       # cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
        #pdb.set_trace()
        for ii in range(len(files)):

            predict_for_image(ii,files=files,data_path=data_path,pred_dir=pred_dir,framestat=framestat,model=model,writer=writer,save_result=True)
            if ii % 100==0:
                print('Video: {0} Done: {1}/{2} images'.format(folder, ii, len(files)))

        myFile = open(os.path.join(local_sub)+folder_name+'.csv', 'w')
        mydet = open(os.path.join(dete_sub)+folder_name+'.csv', 'w')
            #columnTitleRow = "num_frame, detection_output,x_position,y_position, confidence\n"
            #myFile.write(columnTitleRow)
        for row_min in range(0, int((np.shape(framestat)[0])), 5):
            row_max=min(row_min + 5, int((np.shape(framestat)[0])))

            row = framestat[row_min:row_max]
            frame=str(row[0])
            detection=str(row[1])
            x_coord=str(row[2])
            y_coord=str(row[3])
            conf=str(row[4])
            #print(row)
            myFile.write(frame+';'+detection+';'+x_coord+';'+y_coord+';'+ conf+"\n")
            mydet.write(frame+';'+detection+';'+ conf+"\n")
            #csv.write(row)
        myFile.close()
        mydet.close()
        writer.release()