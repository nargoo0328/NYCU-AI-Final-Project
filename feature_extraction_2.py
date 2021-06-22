# first block
import os
import sys
import random
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import keras 
from keras import backend as K
# Root directory of the project
Current_PATH=os.getcwd()
ROOT_DIR =os.path.join(Current_PATH,'Mask_RCNN')
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco
#%matplotlib inline 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
gpu_image=20
#second block
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = gpu_image

config = InferenceConfig()
config.display()

#third block
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)
roi_model=model.create_roi_model()

#fifth block
import cv2
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
label=0
pair_wise_n=100
#batch x 100(frame) x object x4
batch_det=np.zeros((2,100,20,4),dtype=int)
#batch x 1
batch_labels=np.array([],dtype=np.bool_)
#batch x frame x object x 12544
batch_data=np.zeros((2,100,20,12544),dtype=np.float16)
#batch x frame x object x 12544
batch_pair_data=np.zeros((2,100,pair_wise_n,12544),dtype=np.float16)
#batch x 1
batch_ID=np.array([],dtype=np.byte)
batch_num=0
batch_n=1
count=0
for dirr in os.listdir(IMAGE_DIR):
    #training or testing
    if dirr == "training":
      print("HI") 
      continue
    now_path=os.path.join(IMAGE_DIR,dirr)
    ww=os.listdir(now_path)
    ww.sort()
    for d in ww:
        # positive or negative
        print("Now proccessing : ",d)
        if d== "negative":
            label=False
        else:
            label=True
        new_d=os.path.join(now_path,d)
        file_names=os.listdir(new_d)
        file_names.sort()
        for f in file_names:
            #for each video
            start = time.process_time()
            count+=1
            #video ID & label
            idd=os.path.splitext(f)[0]
            print("Now processing : ",idd)
            batch_ID=np.append(batch_ID,[idd],axis=0)
            batch_labels=np.append(batch_labels,[label],axis=0)
            path=os.path.splitext(os.path.join(new_d,f))[0]
            
            # read video
            vidcap = cv2.VideoCapture(os.path.join(new_d,f))
            success,image = vidcap.read()
            temp_frame_single=np.zeros((1,100,20,12544),dtype=np.float16)
            temp_frame_pair=np.zeros((1,100,pair_wise_n,12544),dtype=np.float16)
            batch_det_temp=np.zeros((1,100,20,4),dtype=np.float16)
            frame_num=0
            for ii in range(int(100/gpu_image)):
                images=[]
                for _ in range(gpu_image):
                    if not success:
                        break
                    images.append(image)
                    success,image = vidcap.read()
                start2 = time.time()
                print("Now processing frame: ",frame_num)
                r,detect,p2,p3,p4,p5,image_metas = model.detect(images, verbose=0)
                #for each frame
                object_nums=[]
                for ddd in r:
                    if ddd['class_ids'].shape[0]>=20:
                        object_nums.append(20)
                    else:
                        object_nums.append(ddd['class_ids'].shape[0])
                #start feature extraction
                detect=detect[:,:20,:4]
                detect_out=np.zeros((gpu_image,20,4),dtype=np.float16)
                b=np.zeros((gpu_image,1000,4),dtype=np.float16)
                b[:gpu_image,:20,:]=detect
                b=tf.convert_to_tensor(b,dtype=np.float16)
                #single object
                temp_single=roi_model.predict_on_batch([b, image_metas, p2,p3,p4,p5])
                
                #temp_single=np.float16(temp_single)
                #temp_single =model.detect([image], verbose=0 ,boxes= b,p2=p2,p3=p3,p4=p4,p5=p5,mode=1)
                temp_single=temp_single[:,:20,:,:]
                temp_single=temp_single.reshape(gpu_image,20,12544)
                #frames single
                temp_frame_single[0][frame_num:frame_num+gpu_image]=temp_single
                detect2=np.zeros((gpu_image,pair_wise_n,4),dtype=np.float16)
                for temp,detect_small in enumerate(detect):
                    for i, box in enumerate(detect_small):
                        n_box=model.unmold_new(box,[images[temp]])
                        detect_out[temp,i]=n_box
        
                for index,object_num in enumerate(object_nums):
                    u_count=0
                    h,w=images[index].shape[:2]
                    h/=2
                    w/=2
                    for i in range(object_num):
                        if u_count>=pair_wise_n:
                            break
                        for j in range(i+1,object_num):
                            if (abs(detect_out[index][i][0]-detect_out[index][j][0])>=h or abs(detect_out[index][i][1]-detect_out[index][j][1])>=w or abs(detect_out[index][i][2]-detect_out[index][j][2])>=h or abs(detect_out[index][i][3]-detect_out[index][j][3])>=w) :
                                continue
                            if u_count>=pair_wise_n:
                                break
                            ubox=[min(detect[index][i][0],detect[index][j][0]),min(detect[index][i][1],detect[index][j][1]),max(detect[index][i][2],detect[index][j][2]),max(detect[index][i][3],detect[index][j][3])]
                            ubox=np.array(ubox)
                            detect2[index][u_count]=ubox
                            u_count+=1
                b=np.zeros((gpu_image,1000,4),dtype=np.float16)
                b[:,:pair_wise_n,:]=detect2
                b=tf.convert_to_tensor(b,dtype=np.float16)
                roi =roi_model.predict_on_batch([b, image_metas, p2,p3,p4,p5])
                roi=roi[:,:pair_wise_n]
                roi=roi.reshape(gpu_image,pair_wise_n,12544)
                #frames pair
                temp_frame_pair[0][frame_num:frame_num+gpu_image]=roi

                batch_det_temp[0][frame_num:frame_num+gpu_image]=detect_out
                frame_num+=gpu_image
                print("Finish a frame, time taken: ",time.time() - start2)
            #end of video
            batch_data[batch_num]=temp_frame_single
            batch_pair_data[batch_num]=temp_frame_pair
            batch_det[batch_num]=batch_det_temp
            batch_num+=1
            print("Finish a video, time taken: ",time.process_time() - start)
            #save batch file
            if count==2:
                print("Saveing batch...")
                print("batch num: ",batch_n)
                np.savez('batch_%03d.npz' % batch_n,data=batch_data,labels=batch_labels,ID=batch_ID,det=batch_det,pair_data=batch_pair_data)
                count=0
                batch_num=0
                batch_n+=1
                #batch x 100(frame) x object x4
                batch_det=np.zeros((2,100,20,4),dtype=int)
                #batch x 1
                batch_labels=np.array([],dtype=np.bool_)
                #batch x frame x object x 12544
                batch_data=np.zeros((2,100,20,12544),dtype=np.float16)
                #batch x frame x object x 12544
                batch_pair_data=np.zeros((2,100,pair_wise_n,12544),dtype=np.float16)
                #batch x 1
                batch_ID=np.array([],dtype=np.byte)