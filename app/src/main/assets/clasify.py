import os
import argparse
import keras
import tensorflow as tf
from keras import backend as K
from keras.preprocessing import image
from keras.applications import imagenet_utils
from keras.applications import MobileNet
#from keras.applications import _depthwise_conv_block
#import project modules
import model
from convertToTensorflow import keras_to_tensorflow

#this should be replaced later by the actual model
mobile = keras.applications.mobilenet.MobileNet()
def prepare_image(filename):
    img_path = ''
    img = image.load_img(img_path + filename, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return imagenet_utils.preprocess_input(img_array_expanded_dims)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clasifier options')
    parser.add_argument('--filename', type=str, help='filename of image to classi    fy')
    args = parser.parse_args()
    filename = args.filename
    preprocessed_image = model.prepare_image(filename)
    predictions = mobile.predict(preprocessed_image)
    results = imagenet_utils.decode_predictions(predictions)
    print("Results: ")
    print(results)

