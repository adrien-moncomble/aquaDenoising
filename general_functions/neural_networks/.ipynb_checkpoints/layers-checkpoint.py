'''
Functions with useful layers for the neural networks models.
'''

def conv_batchnorm_relu_block(input_tensor, nb_filter, kernel_size=3, name=None):
    """
    Main construction block of the neural network.
    
    Size transformation : (batch_size, a, b, c) --> (batch_size, a, b, nb_filter)
    
    Parameters:
    input_tensor: Block from which we take the tensor.
    nb_filter: Number of filters we are considering.
    kernel_size: (Default: 3) Size of the nb_filter convolution window we will train to go to the result.
    name: (Default: None) Suffix to identify each blocks in the full model. 
    """
    
    if name is None:
        x = Conv2D(nb_filter, (kernel_size, kernel_size), padding='same')(input_tensor)
        x = BatchNormalization(axis=2)(x)
        x = Activation('relu')(x)
    else:
        x = Conv2D(nb_filter, (kernel_size, kernel_size), padding='same', name=name)(input_tensor)
        x = BatchNormalization(axis=2, name="batchnorm_" + name)(x)
        x = Activation('relu', name="activ_relu_" + name)(x)        

    return x



def SubpixelConv2D(input_shape, scale=4, name=None):
    """
    Keras layer to do subpixel convolution.
    NOTE: Tensorflow backend only. Uses tf.depth_to_space
    Ref:
        [1] Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
            Shi et Al.
            https://arxiv.org/abs/1609.05158

    Parameters:
    input_shape: tensor shape, (batch, height, width, channel)
    name
    scale: upsampling scale. Default=4
    :return:
    """
    # upsample using depth_to_space
    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                input_shape[1] * scale,
                input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        import tensorflow as tf
        return tf.nn.depth_to_space(x, scale)

    if name is None:
        return Lambda(subpixel, output_shape=subpixel_shape)
    else:
        return Lambda(subpixel, output_shape=subpixel_shape, name=name)



def conv2d_block(input_tensor, n_filters, repetition=2, kernel_size=3, batchnorm=True):
    '''
    Function to add blocks of Conv2D, BatchNormalization and a "relu" Activation layer.
    
    Parameters:
    input_tensor: Previous layer in the structure of the neural network.
    n_filters: Depth of the convolutional layers.
    repetition: Number of times this block is repeated
    kernel_size: Size of the kernel used in the convolutional layers.
    batchnorm: If True, adds a BatchNormalization layer after each convolutional layers.
    
    Return:
    x: Final layer of the block.
    '''
    x = input_tensor

    for i in range(repetition):
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
                   kernel_initializer='he_normal', padding='same')(x)
        
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
    
    return x