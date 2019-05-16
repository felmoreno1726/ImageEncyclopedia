import tensorflow
import keras
import numpy as np
import matplotlib.pyplot as plt

#from keras_applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from keras import layers
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout, warnings
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, SeparableConv2D, BatchNormalization
from keras.models import Model
#from keras.engine.topology import get_source_inputs
from keras.applications import imagenet_utils
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input

from model import conv_block, depthwise_conv_block
#from keras.utils import get_file
#from keras.utils import layer_utils
layers=keras.layers

def MobileSqueezeNet(input_shape=(224,224,3), n_classes=1000, depth_multiplier=1, alpha=1.0):
    
    img_input = Input(shape=input_shape)
    
    x = conv_block(img_input, 64, alpha, strides=(2,2))
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='pool1')(x)
    
    x = depthwise_fire_module(x, fire_id=2, squeeze=16, expand=64, depth_multiplier=depth_multiplier)
    x = depthwise_fire_module(x, fire_id=3, squeeze=16, expand=64, depth_multiplier=depth_multiplier)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    x = depthwise_fire_module(x, fire_id=4, squeeze=32, expand=128, depth_multiplier=depth_multiplier)
    x = depthwise_fire_module(x, fire_id=5, squeeze=32, expand=128, depth_multiplier=depth_multiplier)
    #x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

    x = depthwise_fire_module(x, fire_id=6, squeeze=48, expand=192, depth_multiplier=depth_multiplier)
    x = depthwise_fire_module(x, fire_id=7, squeeze=48, expand=192, depth_multiplier=depth_multiplier)
    x = depthwise_fire_module(x, fire_id=8, squeeze=64, expand=256, depth_multiplier=depth_multiplier)
    x = depthwise_fire_module(x, fire_id=9, squeeze=64, expand=256, depth_multiplier=depth_multiplier)
    
    #Header of the imagenet model
    x = Dropout(0.5, name='drop9')(x)
    #x = depthwise_conv_block(x, n_classes, depth_multiplier=depth_multiplier, name='header')
    x = Convolution2D(n_classes, (1, 1), padding='valid', name='conv10')(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax', name='loss')(x)

    model = Model(inputs=img_input, outputs=x, name='mobile_squeezenet')
    return model

def depthwise_fire_module(x, fire_id, squeeze=16, expand=64, depth_multiplier = 1, alpha=1.0):
    s_id = 'fire' + str(fire_id) + '/'
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
    
    x = conv_block(x, squeeze, kernel=(1,1), name=s_id +sq1x1, alpha=alpha)
    print("x: ", x.get_shape)
    left = conv_block(x, expand, name=s_id + exp1x1, alpha=alpha)
    right = depthwise_conv_block(x, expand, name=s_id+exp3x3, depth_multiplier=depth_multiplier, alpha=alpha)
    print("left: ", left.get_shape)
    print("right: ", right.get_shape)
    x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
    return x
def conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1), name=1):
    """Adds an initial convolution layer (with batch normalization and relu6).
    # Arguments
        inputs: Input tensor of shape `(rows, cols, 3)`
            (with `channels_last` data format) or
            (3, rows, cols) (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(224, 224, 3)` would be one valid value.
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution
            along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)`
        if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)`
        if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.
    # Returns
        Output tensor of block.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    if strides == (1, 1):
        x = inputs
    else:
        x = layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv'+str(name)+'_pad')(inputs)
    x = layers.Conv2D(filters, kernel,
                      padding='same',#changed to same
                      use_bias=False,
                      strides=strides,
                      name='conv'+str(name))(x)
    x = layers.BatchNormalization(axis=channel_axis, name='conv'+str(name)+'_bn')(x)
    return layers.ReLU(6., name='conv'+str(name)+'_relu')(x)


def depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), name=1):
    """Adds a depthwise convolution block.
    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu6, pointwise convolution,
    batch normalization and relu6 activation.
    # Arguments
        inputs: Input tensor of shape `(rows, cols, channels)`
            (with `channels_last` data format) or
            (channels, rows, cols) (with `channels_first` data format).
        pointwise_conv_filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the pointwise convolution).
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filters_in * depth_multiplier`.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution
            along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        name: Integer, a unique identification designating
            the block number.
    # Input shape
        4D tensor with shape:
        `(batch, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        4D tensor with shape:
        `(batch, filters, new_rows, new_cols)`
        if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_rows, new_cols, filters)`
        if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.
    # Returns
        Output tensor of block.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = layers.ZeroPadding2D(((0, 1), (0, 1)),
                                 name='conv_pad_%d' % name)(inputs)
    x = layers.DepthwiseConv2D((3, 3),
                               padding='same' if strides == (1, 1) else 'valid',
                               depth_multiplier=depth_multiplier,
                               strides=strides,
                               use_bias=False,
                               name=name+'depthwise_conv' )(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name=name+'depthwise_conv_bn')(x)
    x = layers.ReLU(6., name=name+'depthwise_conv_relu')(x)

    x = layers.Conv2D(pointwise_conv_filters, (1, 1),
                      padding='same',
                      use_bias=False,
                      strides=(1, 1),
                      name=str(name)+ 'pointwise_conv')(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  name=name+'pointwise_conv_bn')(x)
    return layers.ReLU(6., name=name+'pointwise_conv_relu')(x)

