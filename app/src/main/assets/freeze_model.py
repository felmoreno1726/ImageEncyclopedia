import tensorflow as tf
import os
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import *
from convertToTensorflow import *

#this should be replaced later by the model
mobile_model = keras.applications.mobilenet.MobileNet()
#Predict dog image using keras model
output_dir = os.path.join(os.getcwd(), "output_model")

frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in mobile_model.outputs])
tf.train.write_graph(frozen_graph, "./output_model", "mobile_model.pb", as_text=False)
