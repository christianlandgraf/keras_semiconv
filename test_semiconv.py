"""

@author: Christian Landgraf
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input
from keras import initializers
from keras.models import Model

sys.path.append(".")
from keras_semiconv import SemiConv2D, f, rr


def semiconv_model(function=None):
    inputs = Input(shape=(256,256,3))
    
    x = SemiConv2D(3, 3, padding='same', kernel_initializer=initializers.Ones(), 
                   function=function, normalized_position=False)(inputs)
    
    return Model(inputs, x)

if __name__ == '__main__':    
    img = np.ones(shape=(1,256,256,3))
    
    # f(x) = (u,v) - coordinates of pixel x
    model = semiconv_model(f)
    prediction = model.predict(img)
    plt.imshow(prediction[0,:, :, :]/255)
    plt.show()
    
    # f(x) : normalized pixel location
    model = semiconv_model(rr)
    prediction = model.predict(img)
    plt.imshow(prediction[0,:, :, :]/255)
    plt.show()
    
    # TODO: show single channels
