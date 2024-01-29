from tensorflow.keras.utils import Sequence
import tensorflow as tf
import os
import random
from fr_utils import load_file
from config import batch_size, image_shape, embedding_size, file_path
import numpy as np
import albumentations as A


train_file_list = [os.path.join(file_path, entry) for entry in os.listdir(file_path)
                   if entry.startswith(f'train_data_') and entry.endswith('.pkl')]
val_file_list = [os.path.join(file_path, entry) for entry in os.listdir(file_path)
                 if entry.startswith(f'val_data_') and entry.endswith('.pkl')]

transform = A.Compose([
    A.HorizontalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.2),
])


class DataSequence(Sequence):
    def __init__(self, usage):
        self.usage = usage
        if self.usage == 'train':
            print("Loading training data")
            self.samples = train_file_list
        else:
            print("Loading validation data")
            self.samples = val_file_list
            
    def __len__(self):
        return int(len(self.samples)//32)
    
    def __getitem__(self, index):
        i = index * batch_size
        
        length = min(batch_size, (len(self.samples)-1))
        
        anchors = np.zeros((length, image_shape[0], image_shape[1], image_shape[2]))
        positives = np.zeros((length, image_shape[0], image_shape[1], image_shape[2]))
        negatives = np.zeros((length, image_shape[0], image_shape[1], image_shape[2]))
        
        dummy_target = np.zeros((batch_size, embedding_size*3))
        
        for img_num in range(length):
            sample = self.samples[i + img_num]
            
            if self.usage == 'train': 
                anchor = transform(image = np.array(load_file(sample, 'anchor')))
                positive = transform(image = np.array(load_file(sample, 'positive')))
                negative = transform(image = np.array(load_file(sample, 'negative')))
            else:
                anchor = {"image": load_file(sample, 'anchor')}
                positive = {"image": load_file(sample, 'positive')}
                negative = {"image": load_file(sample, 'negative')}
            
            anchors[img_num] = tf.convert_to_tensor(anchor["image"])
            positives[img_num] = tf.convert_to_tensor(positive["image"])
            negatives[img_num] = tf.convert_to_tensor(negative["image"])
            
        data = {
                'anchors': anchors,
                'positives': positives,
                'negatives': negatives
            }   
        return data, dummy_target
    
    def on_epoch_end(self):
        random.shuffle(self.samples)
