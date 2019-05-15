import os
import keras
from keras.models import Model
import tensorflow as tf
from convertToTensorflow import keras_to_tensorflow

#this should be replaced later by the model
mobile_model = keras.applications.mobilenet.MobileNet()
#Predict dog image using keras model
output_dir = os.path.join(os.getcwd(), "output_model")
keras_to_tensorflow(mobile_model, output_dir=output_dir, model_name="mobile_model.pb")

