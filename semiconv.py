"""
An keras implementation of the semi-convolutional operator of [1].

References:
    [1]  D. Novotny, S. Albanie, D. Larlus, A. Vedaldi, “Semi-convolutional
         Operators for Instance Segmentation”,  in ECCV, Sept. 2018.
    [2]  R. Liu, J. Lehman, P. Molino, F. Such, E. Frank, A. Sergeev, and J. Yosinski.,
         "An intriguing failing of convolutional neural networks and the coordconv
         solution". In Advances in Neural Information Processing Systems, 2018,
         pp. 9628-9639.
    [3]  "keras-coordconv", https://github.com/titu1994/keras-coordconv/.


@author: Christian Landgraf
"""

import keras.backend as K
from keras.layers import Layer
from keras import initializers, regularizers, constraints, activations
from keras.utils import conv_utils
from keras.engine.base_layer import InputSpec


class SemiConv2D(Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 function=None,
                 arguments=None,
                 normalized_position=True,
                 **kwargs):
        super(SemiConv2D, self).__init__(**kwargs)
        self.rank = 2
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, self.rank,
                                                      'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, self.rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, self.rank,
                                                        'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank+2)
        self.function = function
        self.arguments = arguments if arguments else {}        
        self.normalized_position = normalized_position

    def build(self, input_shape):
        
        #Ordinary Conv2D kernel
        channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        
        #Ordinary Conv2D Convolution kernel
        outputs = K.conv2d(
            inputs,
            self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format='channels_last',
            dilation_rate=self.dilation_rate)


        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format='channels_last')

        if self.activation is not None:
            outputs = self.activation(outputs)
        
        #Add second part of semi-convolutional operator
        shape = K.shape(outputs)
        shape = [shape[i] for i in range(4)]
        batch_size, x_dim, y_dim, c1 = shape
        
        #Create tensors containng x/y pixel locations
        xx_ones = K.ones([batch_size, x_dim], dtype='int32')
        xx_ones = K.expand_dims(xx_ones, -1)
        xx_range = K.tile(K.expand_dims(K.arange(x_dim), 0), [batch_size, 1])
        xx_range = K.expand_dims(xx_range, 1)
        xx_channel = K.dot(xx_ones, xx_range)
        xx_channel = K.expand_dims(xx_channel, -1)
        xx_channel = K.cast(xx_channel, 'float32')
        if self.normalized_position:
            xx_channel = xx_channel / (K.cast(x_dim, 'float32') - 1)
            xx_channel = xx_channel*2 - 1
        
        yy_ones = K.ones([batch_size, y_dim], dtype='int32')
        yy_ones = K.expand_dims(yy_ones, 1)
        yy_range = K.tile(K.expand_dims(K.arange(y_dim), 0), [batch_size, 1])
        yy_range = K.expand_dims(yy_range, -1)
        yy_channel = K.dot(yy_range, yy_ones)
        yy_channel = K.expand_dims(yy_channel, -1)
        yy_channel = K.cast(yy_channel, 'float32')
        if self.normalized_position:
            yy_channel = yy_channel / (K.cast(x_dim, 'float32') - 1)
            yy_channel = yy_channel*2 - 1
        
        #Concat global x and y location
        semi_tensor = K.squeeze(K.concatenate([xx_channel,yy_channel], axis=-1), axis=2)

        #Apply Lambda function
        if self.function is not None:
            semi_tensor = self.function(semi_tensor,self.normalized_position,**self.arguments)
        
        c2 = K.shape(semi_tensor)[-1]
            
        #Pad with "zero" channels
        semi_tensor = K.concatenate([semi_tensor,K.zeros([batch_size, x_dim, y_dim, c1-c2])], axis=-1)        
        
        #Sum the convolutional output with the semi_tensor
        joint_outputs = outputs + semi_tensor
        return joint_outputs#, semi_tensor, outputs
    

    def compute_output_shape(self, input_shape):
         space = input_shape[1:-1]
         new_space = []
         for i in range(len(space)):
             new_dim = conv_utils.conv_output_length(
                 space[i],
                 self.kernel_size[i],
                 padding=self.padding,
                 stride=self.strides[i],
                 dilation=self.dilation_rate[i])
             new_space.append(new_dim)
         return (input_shape[0],) + tuple(new_space) + (self.filters,)

    def get_config(self):
        config = {
            'rank': self.rank,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'function': self.function,
            'arguments': self.arguments,
            'normalized_position': self.normalized_position
        }
        base_config = super(SemiConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

################################################################################
# Kernel for pixel embedding
################################################################################

# TODO 

################################################################################
#Example Functions:
################################################################################

#Multiplies each entry by a constant
#f(x) = a*x
def f(xy, normalized_position):
    xy_new = 1 * xy
    return xy_new

#Merge the two feature maps into one, following the CoordConv paper
def rr(xy, normalized_position):
    if normalized_position:
        rr = K.sqrt(K.square(xy[:,:,:,0]-0.5) + K.square(xy[:,:,:,1]-0.5))
        rr = K.expand_dims(rr, axis=-1)
    else:
        x_dim = K.shape(xy)[1]
        y_dim = K.shape(xy)[2]
        rr = K.sqrt(K.square(xy[:,:,:,0]-(K.cast(x_dim, 'float32')/K.constant(2.0))) + 
                     K.square(xy[:,:,:,1]-(K.cast(y_dim, 'float32')/K.constant(2.0))))
        rr = K.expand_dims(rr, axis=-1)
        
    return rr

 