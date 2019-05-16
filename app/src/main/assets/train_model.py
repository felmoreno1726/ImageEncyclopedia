import os
import tensorflow as tf
import keras
from keras import backend as K
import keras
from keras.models import Model

from convertToTensorflow import *

#this should be replaced later by the model
mobile_model = keras.applications.mobilenet.MobileNet()
#Predict dog image using keras model
output_dir = os.path.join(os.getcwd(), "keras_weights/")
mobile_model.save_weights(output_dir + "mobilenet.h5")
#keras_to_tensorflow(mobile_model, output_dir=output_dir, model_name="mobile_model.pb")