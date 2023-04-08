#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %matplotlib inline
import os
import sys
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import os
# Create a MirroredStrateg, If Multi-GPU available
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' 
tf.config.set_soft_device_placement(True)
# strategy = tf.distribute.MirroredStrategy(devices=['/gpu:1','/gpu:2']) 
# print('Number of GPUs being used: {}'.format(strategy.num_replicas_in_sync))
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.utils import to_categorical

from livelossplot.keras import PlotLossesCallback
from livelossplot import PlotLossesKerasTF
import efficientnet.keras as efn
import h5py, glob, re, cv2, math, matplotlib
import tensorflow.keras as keras
import pandas as pd
from pandas import read_csv
import numpy as np
from scipy import stats
import itertools, random

# from cnn_utils import *
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from PIL import Image  
import pdb
from statistics import mode 
from IPython.display import clear_output

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

#np.random.seed(1)
## Setting the seed for python random numbers
#random.seed(1254)
## Setting the graph-level random seed.
#tf.random.set_seed(89)

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#get_ipython().system('CUDA_VISIBLE_DEVICES=2')

# strategy = tf.distribute.MirroredStrategy(devices=['/gpu:1','/gpu:2'])
# print('Number of GPUs being used: {}'.format(strategy.num_replicas_in_sync))
# print('Number of GPUs being used: {}'.format(strategy.num_replicas_in_sync))
# def setup_multi_node_training(): # IMPORTANT: SET UP TF_CONFIG FOR MULTINODE TRAINING HERE os.environ[“TF_FORCE_GPU_ALLOW_GROWTH”] = “true” tf.config.set_soft_device_placement(True) mirrored_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(tf.distribute.experimental.CollectiveCommunication.NCCL) # Constructs the configuration run_config = tf.estimator.RunConfig( train_distribute=mirrored_strategy, ) return run_config
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# ## Load Data

# In[2]:

def main():
	#filename = '/home/radar/seqeuntial-model/All_subjects_RDmap_jetson.hdf5'

	#data = h5py.File(filename, "r")
	#print('Selected File: '+str(filename))
	#	x_train1 = np.array(data["train_img"])
	#	y_train1 = np.array(data["train_labels"])
	#x_test1 = np.array(data["test_img"])
	#y_test1 = np.array(data["test_labels"])
	#data.close()
	#	print(x_train1.shape)
	#	print(y_train1.shape)
	#print(x_test1.shape)
	#print(y_test1.shape)


	# filename is the png spectrogram created during the signal processing step
	filename = '/home/radar/seqeuntial-model/SLEEP.png'
	img = Image.open(filename)
	print('step before img.size')
	width, height = img.size
	print('width:')
	# ratio is the length of 1 second, found by dividing the width by the total number of seconds collected for (3.8s for BOOK)
	ratio = width // 3.6
	print('ratio:')
	# w is the size for a 0.2 second window found by multiplying the length of a 1s interval by 0.2
	w = str(ratio * 0.2)
	print('w:' + w)
	w = round(float(w))
	w = int(w)

	frame_num = 1
	for col_i in range(0,width,w):
		crop = img.crop((col_i, 0, col_i + w, height))
		#save_to = os.isfile(os.path.join(savedir, "cropped_Book_" + str(frame_num) + ".png"))
		#crop.save(save_to.format(frame_num))
		crop.save('/home/radar/seqeuntial-model/test_spec/sleep/cropped_Sleep_' + str(frame_num) + '.png')
		frame_num += 1

	X_data = []
	files = sorted(glob.glob ('/home/radar/seqeuntial-model/test_spec/sleep/*.png'))
	for myFile in files:
		print(myFile)
		image = cv2.imread(myFile)
		image = cv2.resize(image, (128,128), interpolation = cv2.INTER_CUBIC)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		X_data.append (image)

	print("X_data shape:", np.array(X_data).shape)

	x_test1 = np.expand_dims(np.array(X_data),0)
	x_test1 = x_test1/255.

	print("number of test examples = " +str(x_test1.shape[0]))
	print("x_test1 shape: " + str(x_test1.shape))

		# In[4]:


	#	# windowed RD
	#	interval = range(0,600)
	#	x_train11 = np.reshape(x_train1[:,interval,:,:,:], (x_train1.shape[0],120,5,x_train1.shape[2],x_train1.shape[3],x_train1.shape[4]))
	#	x_test11 = np.reshape(x_test1[:,interval,:,:,:], (x_test1.shape[0],120,5,x_test1.shape[2],x_test1.shape[3],x_test1.shape[4]))
	#	y_train11 = np.reshape(np.argmax(y_train1[:,interval,:],-1), (y_train1.shape[0],120,5))
	#	y_test11 = np.reshape(np.argmax(y_test1[:,interval,:],-1), (y_test1.shape[0],120,5))
	#	y_train11 = to_categorical(np.squeeze(stats.mode(y_train11,2)[0]))
	#	y_test11 = to_categorical(np.squeeze(stats.mode(y_test11,2)[0]))
	#	print(x_train11.shape)
	#	print(y_train11.shape)
	#	print(x_test11.shape)
	#	print(y_test11.shape)
	#
	#
	#	duration = 24.2
	#	nsta_sec = 0.7
	#	ratio = margin_tr.shape[-1]/duration
	#	nsta = int(nsta_sec*ratio)
	#	nlta = int(2*nsta)
	#	stepsz = int(0.2*ratio) # 0.2
	#	timevec = np.linspace(0,24.2,margin_ts.shape[1])
	#	idx = 27
	#	init_th = 0.6
	#	stop_th = 0.3
	#	vecs_train = [] #np.zeros(np.argmax(margin_tr,-1).shape)
	#	mask_train = [] #np.zeros(np.argmax(margin_tr,-1).shape)
	#	vecs_test = [] #np.zeros(np.argmax(margin_ts,-1).shape)
	#	mask_test = [] #np.zeros(np.argmax(margin_ts,-1).shape)
	#	for i in range(len(margin_tr)):
	#	    vecs_train.append(sta_lta2(margin_tr[i],nlta,nsta,init_th,stop_th,stepsz)[0])
	#	    mask_train.append(sta_lta2(margin_tr[i],nlta,nsta,init_th,stop_th,stepsz)[1])
	#	# mask_train_win = mask_train[:,::5]
	#	for i in range(len(margin_ts)):
	#	    vecs_test.append(sta_lta2(margin_ts[i],nlta,nsta,init_th,stop_th,stepsz)[0])
	#	    mask_test.append(sta_lta2(margin_ts[i],nlta,nsta,init_th,stop_th,stepsz)[1])
	#	# mask_test_win = mask_test[:,::5]
	#	mask_train = np.array(mask_train)
	#	mask_test = np.array(mask_test)
	#
	#	# ### Visualize
	#
	#	# In[10]:
	#
	#
	#	np.unique(np.argmax(y_test3[idx],-1))
	#
	#	# In[11]:
	#
	#
	#	# downsample the mask to vid
	#	mask_vid_tr = np.zeros(np.argmax(y_train1,-1).shape)
	#	ratio1 = mask_train.shape[-1]/mask_vid_tr.shape[-1]
	#	for i in range(len(mask_train)):
	#	    for j in range(mask_train.shape[-1]):
	#		if mask_train[i,j] == 0:
	#		    continue
	#		else:
	#		    mask_vid_tr[i,int(j/ratio1)] = 1
	#	mask_vid_ts = np.zeros(np.argmax(y_test1,-1).shape)
	#	for i in range(len(mask_test)):
	#	    for j in range(mask_test.shape[-1]):
	#		if mask_test[i,j] == 0:
	#		    continue
	#		else:
	#		    mask_vid_ts[i,int(j/ratio1)] = 1

	#	# downsample the mask to spect
	#	mask_spect_tr = np.zeros(np.argmax(y_train3,-1).shape)
	#	ratio2 = mask_train.shape[-1]/mask_spect_tr.shape[-1]
	#	for i in range(len(mask_train)):
	#	    for j in range(mask_train.shape[-1]):
	#		if mask_train[i,j] == 0:
	#		    continue
	#		else:
	#		    mask_spect_tr[i,int(j/ratio2)] = 1
	#	mask_spect_ts = np.zeros(np.argmax(y_test3,-1).shape)
	#	for i in range(len(mask_test)):
	#	    for j in range(mask_test.shape[-1]):
	#		if mask_test[i,j] == 0:
	#		    continue
	#		else:
	#		    mask_spect_ts[i,int(j/ratio2)] = 1
	#	print('Mask shapes:')
	#	print(mask_vid_tr.shape)
	#	print(mask_vid_ts.shape)
	#	print(mask_spect_tr.shape)
	#	print(mask_spect_ts.shape)
	#
	#	numgest = [gesture_counter(m) for m in mask_spect_ts]
	#	sum(numgest)
	#
	#
	#	# In[14]:
	#
	#
	#	num_class = y_test3.shape[2]
	#	num_class
	#
	#	x1train, y1train = masker(x_train11, y_train3, mask_spect_tr)
	#	x1test, y1test = masker(x_test11, y_test3, mask_spect_ts)
	#	x2train, y2train = masker(x_train12, y_train3, mask_spect_tr)
	#	x2test, y2test = masker(x_test12, y_test3, mask_spect_ts)
	#	x3train, y3train = masker(x_train3, y_train3, mask_spect_tr)
	#	x3test, y3test = masker(x_test3, y_test3, mask_spect_ts)
	#	print(x1train.shape)
	#	print(y1train.shape)
	#	print(x1test.shape)
	#	print(y1test.shape)
	#	print(x2train.shape)
	#	print(y2train.shape)
	#	print(x2test.shape)
	#	print(y2test.shape)
	#	print(x3train.shape)
	#	print(y3train.shape)
	#	print(x3test.shape)
	#	print(y3test.shape)
	#
	#
	#	# In[56]:
	#
	#
	#	x1test_nowin, y1test_nowin = masker(x_test1, y_test1, mask_vid_ts)
	#	x2test_nowin, y2test_nowin = masker(x_test2, y_test2, mask_vid_ts)
	#	print(x1test_nowin.shape)
	#	print(y1test_nowin.shape)
	#	print(x2test_nowin.shape)
	#	print(y2test_nowin.shape)
	#

		# In[19]:


	model_file = 'Models/ctc model md.json'
	w_file = 'Models/ctc model md.h5'
	json_file = open(model_file, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model_nogd_rd = model_from_json(loaded_model_json)
	model_nogd_rd.load_weights(w_file)


		# In[68]:


		#predict_generator = data_generator_nolabel(x2test_nowin)
	predict_generator = data_generator_nolabel(x_test1)


		# In[63]:


		#pred = [np.argmax(np.squeeze(model_nogd_ra.predict(next(predict_generator))),-1) for i in range(len(x2test_nowin))]
	pred = [np.argmax(np.squeeze(model_nogd_rd.predict(next(predict_generator))),-1) for i in range(len(x_test1))]


		# In[64]:


		# best path decoding
	pred_labels = np.squeeze(np.array([stats.mode(p)[0] for p in pred]))
	pred_labels = np.squeeze(np.array([p for p in pred]))
	pred_labels = pred_labels.flatten()

		# In[65]:


		#actuals = np.squeeze(np.array([stats.mode(np.argmax(y,-1))[0] for y in y2test_nowin]))
#	actuals = np.squeeze(np.array([stats.mode(np.argmax(y,-1))[0] for y in y_test1]))
#	actuals = np.squeeze(np.array([np.argmax(y,-1) for y in y_test1]))
#	actuals = actuals.flatten()

		# In[66]:


#	cmp = pred_labels == actuals
#	print(cmp)
#	print(pred_labels.shape)
#	print(actuals.shape)
#	trues = np.sum(cmp)
#	acc = trues/len(cmp)*100
#	acc
#	print(acc)

	#sign_class = np.argmax(pred_labels,axis=1)
	#sign_class = np.argmax(pred_labels)
	sign_class = max(pred_labels)
	print(pred_labels)
	print(sign_class)

	if sign_class == [1]: print("Walking")
	elif sign_class == [2]: print("Sitting")
	elif sign_class == [3]: print("Standing up")
	elif sign_class == [4]: print("tired")
	elif sign_class == [5]: print("book")
	elif sign_class == [6]: print("sleep")
	elif sign_class == [7]: print("evening")
	elif sign_class == [8]: print("ready")
	elif sign_class == [9]: print("hot")
	elif sign_class == [10]: print("month")
	elif sign_class == [11]: print("cook")
	elif sign_class == [12]: print("again")
	elif sign_class == [13]: print("summon")
	elif sign_class == [14]: print("maybe")
	elif sign_class == [15]: print("night")
	elif sign_class == [16]: print("something")
	elif sign_class == [17]: print("teacher")
	elif sign_class == [18]: print("teach")




#def sta_lta2(vec,nlta,nsta,init_th,stop_th,stepsz):
#    vec2 = np.zeros(vec.shape)
#    mask = np.zeros(vec.shape)
#    state = 0 # '0' nothing, '1' signing
#    
#    for i in range(0,len(vec),stepsz):
#        
#        if i+nlta+nsta+1 > len(vec):
#            if state == 1:
#                stoppt = len(vec)-2
#                vec2[startpt:stoppt] = vec[startpt:stoppt]
#                mask[startpt:stoppt] = 1
#            break
#            
#            
#        longwin = vec[i:i+nlta]
#        shortwin = vec[i+nlta:i+nlta+nsta]
#        
#        if i < nlta and np.mean(longwin) > 150:
#            vec2[0:i+nsta] = vec[0:i+nsta]
#            mask[0:i+nsta] = 1
#        if init_th < sum(shortwin)/sum(longwin):
#            if state == 0:
#                startpt = i+nlta
#                state = 1
#            if state == 1:
#                continue
#        
#        else:
#            if state == 0:
#                continue
#            if state == 1:
#                if sum(shortwin)/sum(longwin) > stop_th:
#                    continue
#                else:
#                    stoppt = i+nlta+int(nsta/2)
#                    state = 0
#                    vec2[startpt:stoppt] = vec[startpt:stoppt]
#                    mask[startpt:stoppt] = 1
#                    
#    return vec2, mask


# In[9]:


	




	

# In[12]:


#def gesture_counter(mask):
#    cnt = 0
#    flag = 0
#    for i in range(len(mask)):
#        if flag == 0 and mask[i] == 0:
#            continue
#        if flag == 1 and mask[i] == 1:
#            continue
#        if flag == 0 and mask[i] == 1:
#            flag = 1
#            cnt += 1
#        if flag == 1 and mask[i] == 0:
#            flag = 0
#    return cnt


# In[54]:




# In[15]:


#def masker(x, y, mask):
#    x2 = []
#    y2 = []
#    flag = 0
#    for i in range(len(mask)):
#        for j in range(mask.shape[1]):
#            if flag == 0 and mask[i,j] == 0:
#                continue
#            if flag == 1 and mask[i,j] == 1:
#                if j == mask.shape[1]-1 and j+1 - startpt > 2:
#                    stoppt = j+1
#                    x2.append(x[i,startpt:stoppt])
#                    y2.append(np.squeeze(np.argmax(y[i,startpt:stoppt],-1)))
#                else:
#                    continue
#            if flag == 0 and mask[i,j] == 1:
#                flag = 1
#                startpt = j
#            if flag == 1 and mask[i,j] == 0:
#                flag = 0
#                stoppt = j+1
#                if stoppt-startpt > 2:
#                    x2.append(x[i,startpt:stoppt])
#                    y2.append(np.squeeze(np.argmax(y[i,startpt:stoppt],-1)))
#    
#    x2 = np.asarray(x2)
#    y2 = np.array([to_categorical(y,num_classes=num_class) for y in np.array(y2)])
#    return x2, y2


# In[16]:




# ### No GD

# In[21]:


def data_generator_nolabel(data, batch_size=1):              
    """
    Yields the next training batch.
    data is an array  [[[frame1_filename,frame2_filename,…frame16_filename],label1], [[frame1_filename,frame2_filename,…frame16_filename],label2],……….].
    """
    num_samples = data.shape[0]

    while True:
        for offset in range(0, num_samples, batch_size):
    #             print ('starting index: ', offset) 
            # Get the samples you'll use in this batch
            batch_samples = data[offset:offset+batch_size]
#            label = labels[offset:offset+batch_size]
            # Initialise X_train and y_train arrays for this batch
            X_train = []
#            y_train = []
            # For each example
            for i in range(0,batch_samples.shape[0]):
                X_train.append(batch_samples[i])
#                 y_train.append(np.array([ord(y)%32 for y in label[i]]))

            # Make sure they're numpy arrays (as opposed to lists)
            X_train = np.array(X_train)
            #X_train = np.rollaxis(X_train,1,4)
#             y_train = np.array(y_train)

            # yield the next training batch
            yield X_train


# In[17]:


# In[ ]:


if __name__ == '__main__':
    main()

