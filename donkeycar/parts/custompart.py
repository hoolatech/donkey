### Generic pilot reference for donkey car 

# @author: Sunil Mallya

# Import your DeepLearning framework
#import mxnet as mx
#from mxnet import gluon
#import keras as K
#import torch.nn as nn
#import tensorflow as tf

import os
import numpy as np
import logging
from PIL import Image

head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.INFO, format=head)
    
def get_train_data_numpy(df, batch):
    '''
    @return: image numpy array, angle values, throttle values
    '''
    images = []
    a_labels = []
    t_labels = []

    for index, row in df.iterrows():
        img = np.array(Image.open(row['cam/image_array'])) 
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        images.append(img)
        a_labels.append(row['user/angle'])
        t_labels.append(row['user/throttle'])

    np_images = np.stack(images)    
    a_labels = np.stack(a_labels)
    t_labels = np.stack(t_labels)
    return np_images, a_labels, t_labels

#### PILOT CLASSES ####

class BasePilot():
    '''
    Any pilot you create must override the run function and should use the following two functions
    1. Load: Loaded previously trained model
    2. Train: Train the chosen model architecture
    '''

    @classmethod
    def default_model(cls):
        # TODO: build your basic model architecture here
        pass
    
    def load(self, model_path):
        '''
        Load a pretrained model
        '''
        #self.model = load_model()
        pass

    def train(self, train_iter, val_iter, 
              saved_model_path, num_epoch=100, steps=100, train_split=0.8,
              verbose=1, min_delta=.0005, patience=5, use_early_stop=True):

        '''
        Training code
        
        For angle prediction, MAE should be sufficient, but do explore other metrics
        '''
        # TODO: Write your training loop here
        pass

    def run(self, img_arr):
        '''
        Run handler, the donkey framework will send numpy image arr for inference

        @return float <Steeting Angle>, float <Throttle>  
        '''
        #TODO: Override this function in your dervice pilot class
        pass

    def get_train_val_iter(self, train_df, val_df, batch_size):
        '''
        This function is called in the donkey2.py when training is initiated
        '''
        #TODO: Transform your data in to numpy or iterators that train code consumes
        
        train_iter = get_train_data_numpy(train_df, batch_size)
        val_iter = get_train_data_numpy(val_df, batch_size)
        return train_iter, val_iter


class LinearPilot(BasePilot):
    '''
    Pure regression model 
    '''
    def __init__(self, model=None, num_outputs=None, *args, **kwargs):
        
        super(LinearPilot, self).__init__(*args, **kwargs)
        if model:
            self.model = model
        else:
            self.model = BasePilot.default_model()

    def run(self, img_arr):
        # switch channels and add expand dimensions to include batch_size
        img_arr = np.swapaxes(img_arr, 0, 2)
        img_arr = np.swapaxes(img_arr, 1, 2)
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        
        #self.model.forward(img_arr)
        #outputs = self.model.get_outputs()
        #steering = outputs[0].asnumpy()
        #throttle = outputs[1].asnumpy()

        return steering[0][0], throttle[0][0]

class CategoricalPilot(BasePilot):
    '''
    Worth trying to bucket the angles rather than a pure regression
    '''
    pass

def get_my_pilot():
    '''
    get the pilot of choice
    '''
    return LinearPilot() 
