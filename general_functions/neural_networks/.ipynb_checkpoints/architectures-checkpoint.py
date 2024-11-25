'''
Functions to create different neural network architectures.
'''

def model_PDHMnet(inputs, n_labels, dropout=0, using_deep_supervision=False):
    """
    Model to create a Deep P-DHM Net.
    
    inputs: Input layer of the model.
    n_labels: Numbers of output layers requiered.
    dropout: Rate of dropout at each dropout layer.
    using_deep_supervision: Decouple all outputs if True.
    """
    
    # Size of each layers
    nb_filter = [64, 64, 64, 64, 64]

    # Set image data format to channels first
    global bn_axis
    K.set_image_data_format("channels_last")
    bn_axis = -1
    
    ##### 1-1
    ## Input : [inputs]
    ## Outputs : [2-1, 1-2, 1-3, 1-4, 1-5]
    ## Transform : (batch_size, a, b, c) ---> (batch_size, a/2, b/2, nb_filter[0])
    conv1_1 = conv_batchnorm_relu_block(inputs, nb_filter=nb_filter[0], name="conv2d_1-1")
    pool1_1 = MaxPooling2D((2, 2), name="maxpooling_1-1")(conv1_1)
    pool1_1 = Dropout(dropout, name="dropout_1-1")(pool1_1)
    
    ##### 2-1
    ## Input : [1-1]
    ## Outputs : [3-1, 2-2, 2-3, 2-4, 1-2]
    ## Transform : (batch_size, a/2, b/2, nb_filter[0]) ---> (batch_size, a/4, b/4, nb_filter[1])
    conv2_1 = conv_batchnorm_relu_block(pool1_1, nb_filter=nb_filter[1], name="conv2d_2-1")
    pool2_1 = MaxPooling2D((2, 2), name="maxpooling_2-1")(conv2_1)
    pool2_1 = Dropout(dropout, name="dropout_2-1")(pool2_1)
    
    ##### 3-1
    ## Input : [2-1]
    ## Outputs : [4-1, 3-2, 3-3, 2-2]
    ## Transform : (batch_size, a/4, b/4, nb_filter[1]) ---> (batch_size, a/8, b/8, nb_filter[2])
    conv3_1 = conv_batchnorm_relu_block(pool2_1, nb_filter=nb_filter[2], name="conv2d_3-1")
    pool3_1 = MaxPooling2D((2, 2), name="maxpooling_3-1")(conv3_1)
    pool3_1 = Dropout(dropout, name="dropout_3-1")(pool3_1)
    
    ##### 4-1
    ## Input : [3-1]
    ## Outputs : [5-1, 4-2, 3-2]
    ## Transform : (batch_size, a/8, b/8, nb_filter[2]) ---> (batch_size, a/16, b/16, nb_filter[3])
    conv4_1 = conv_batchnorm_relu_block(pool3_1, nb_filter=nb_filter[3], name="conv2d_4-1")
    pool4_1 = MaxPooling2D((2, 2), name="maxpooling_4-1")(conv4_1)
    pool4_1 = Dropout(dropout, name="dropout_4-1")(pool4_1)
    
    ##### 5-1
    ## Input : [4-1]
    ## Outputs : [4-2]
    ## Transform : (batch_size, a/16, b/16, nb_filter[3]) ---> (batch_size, a/32, b/32, nb_filter[4])
    conv5_1 = conv_batchnorm_relu_block(pool4_1, nb_filter=nb_filter[4], name="conv2d_5-1")
    conv5_1 = Dropout(dropout, name="dropout_5-1")(conv5_1)
    
    ##### 1-2
    ## Input : [2-1, 1-1]
    ## Outputs : [1-3, 1-4, 1-5, outputs]
    ## Transform : (batch_size, a, b, nb_filter[1]/4 + nb_filter[0]) ---> (batch_size, a, b, nb_filter[0])
    up1_2 = SubpixelConv2D(conv2_1, scale=2, name='up_1-2')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge_1-2', axis=bn_axis)
    conv1_2 = conv_batchnorm_relu_block(conv1_2,  nb_filter=nb_filter[0], name="conv2d_1-2")
    conv1_2 = Dropout(dropout, name="dropout_1-2")(conv1_2)
    
    ##### 2-2
    ## Input : [3-1, 2-1]
    ## Outputs : [2-3, 2-4, 1-3]
    ## Transform : (batch_size, a/2, b/2, nb_filter[2]/4 + nb_filter[1]) ---> (batch_size, a/2, b/2, nb_filter[1])
    up2_2 = SubpixelConv2D(conv3_1,'up_2-2',scale=2)(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge_2-2', axis=bn_axis)
    conv2_2 = conv_batchnorm_relu_block(conv2_2,  nb_filter=nb_filter[1], name="conv2d_2-2")
    conv2_2 = Dropout(dropout, name="dropout_2-2")(conv2_2)
    
    ##### 3-2
    ## Input : [4-1, 3-1]
    ## Outputs : [3-3, 2-3]
    ## Transform : (batch_size, a/4, b/4, nb_filter[3]/4 + nb_filter[2]) ---> (batch_size, a/4, b/4, nb_filter[2])
    up3_2 = SubpixelConv2D(conv4_1,'up_3-2',scale=2)(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge_3-2', axis=bn_axis)
    conv3_2 = conv_batchnorm_relu_block(conv3_2,  nb_filter=nb_filter[2], name="conv2d_3-2")
    conv3_2 = Dropout(dropout, name="dropout_3-2")(conv3_2)
    
    ##### 4-2
    ## Input : [5-1, 4-1]
    ## Outputs : [3-3]
    ## Transform : (batch_size, a/8, b/8, nb_filter[4]/4 + nb_filter[3]) ---> (batch_size, a/8, b/8, nb_filter[3])
    up4_2 = SubpixelConv2D(conv5_1,'up_4-2',scale=2)(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge_4-2', axis=bn_axis)
    conv4_2 = conv_batchnorm_relu_block(conv4_2,  nb_filter=nb_filter[3], name="conv2d_4-2")
    conv4_2 = Dropout(dropout, name="dropout_4-2")(conv4_2)
    
    ##### 1-3
    ## Input : [2-2, 1-1, 1-2]
    ## Outputs : [1-4, 1-5, outputs]
    ## Transform : (batch_size, a, b, nb_filter[1]/4 + nb_filter[0] + nb_filter[0]) ---> (batch_size, a, b, nb_filter[0])
    up1_3 = SubpixelConv2D(conv2_2,'up_1-3',scale=2)(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge_1-3', axis=bn_axis)
    conv1_3 = conv_batchnorm_relu_block(conv1_3,  nb_filter=nb_filter[0], name="conv2d_1-3")
    conv1_3 = Dropout(dropout, name="dropout_1-3")(conv1_3)
    
    ##### 2-3
    ## Input : [3-2, 2-1, 2-2]
    ## Outputs : [2-4, 1-4]
    ## Transform : (batch_size, a/2, b/2, nb_filter[2]/4 + nb_filter[1] + nb_filter[1]) ---> (batch_size, a/2, b/2, nb_filter[1])
    up2_3 = SubpixelConv2D(conv3_2,'up_2-3',scale=2)(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge_2-3', axis=bn_axis)
    conv2_3 = conv_batchnorm_relu_block(conv2_3,  nb_filter=nb_filter[1], name="conv2d_2-3")
    conv2_3 = Dropout(dropout, name="dropout_2-3")(conv2_3)
    
    ##### 3-3
    ## Input : [4-2, 3-1, 3-2]
    ## Outputs : [2-4]
    ## Transform : (batch_size, a/4, b/4, nb_filter[3]/4 + nb_filter[2] + nb_filter[2]) ---> (batch_size, a/4, b/4, nb_filter[2])
    up3_3 = SubpixelConv2D(conv4_2,'up_3-3',scale=2)(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge_3-3', axis=bn_axis)
    conv3_3 = conv_batchnorm_relu_block(conv3_3,  nb_filter=nb_filter[2], name="conv2d_3-3")
    conv3_3 = Dropout(dropout, name="dropout_3-3")(conv3_3)
    
    ##### 1-4
    ## Input : [2-3, 1-1, 1-2, 1-3]
    ## Outputs : [1-5, outputs]
    ## Transform : (batch_size, a, b, nb_filter[1]/4 + nb_filter[0] + nb_filter[0] + nb_filter[0]) ---> (batch_size, a, b, nb_filter[0])
    up1_4 = SubpixelConv2D(conv2_3,'up_1-4',scale=2)(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge_1-4', axis=bn_axis)
    conv1_4 = conv_batchnorm_relu_block(conv1_4,  nb_filter=nb_filter[0], name="conv2d_1-4")
    conv1_4 = Dropout(dropout, name="dropout_1-4")(conv1_4)
    
    ##### 2-4
    ## Input : [3-3, 2-1, 2-2, 2-3]
    ## Outputs : [1-5]
    ## Transform : (batch_size, a/2, b/2, nb_filter[2]/4 + nb_filter[1] + nb_filter[1] + nb_filter[1]) ---> (batch_size, a/2, b/2, nb_filter[1])
    up2_4 = SubpixelConv2D(conv3_3,'up_2-4',scale=2)(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge_2-4', axis=bn_axis)
    conv2_4 = conv_batchnorm_relu_block(conv2_4,  nb_filter=nb_filter[1], name="conv2d_2-4")
    conv2_4 = Dropout(dropout, name="dropout_2-4")(conv2_4)
    
    ##### 1-5
    ## Input : [2-4, 1-1, 1-2, 1-3, 1-4]
    ## Outputs : [outputs]
    ## Transform : (batch_size, a, b, nb_filter[1]/4 + nb_filter[0] + nb_filter[0] + nb_filter[0] + nb_filter[0]) ---> (batch_size, a, b, nb_filter[0])
    up1_5 = SubpixelConv2D(conv2_4,'up_1-5',scale=2)(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge_1-5', axis=bn_axis)
    conv1_5 = conv_batchnorm_relu_block(conv1_5,  nb_filter=nb_filter[0], name="conv2d_1-5")
    conv1_5 = Dropout(dropout, name="dropout_1-5")(conv1_5)   
    
    ##### outputs
    ## Inputs : [1-2, 1-3, 1-4, 1-5, 1-6]
    nestnet_output_1 = Conv2D(n_labels, (1, 1), activation='linear', name='output_1',padding='same')(conv1_2)
    nestnet_output_2 = Conv2D(n_labels, (1, 1), activation='linear', name='output_2', padding='same' )(conv1_3)
    nestnet_output_3 = Conv2D(n_labels, (1, 1), activation='linear', name='output_3', padding='same')(conv1_4)
    nestnet_output_4 = Conv2D(n_labels, (1, 1), activation='linear', name='output_4', padding='same')(conv1_5)
    
    conv1_6 = concatenate([nestnet_output_1,nestnet_output_2, nestnet_output_3, nestnet_output_4], name='merge_1-6', axis=bn_axis)
    nestnet_output_denoise = Conv2D(n_labels, (1, 1), activation='linear', name='output_denoised',padding='same')(conv1_6)
    
    if using_deep_supervision:
        model = Model(input=inputs, output=[nestnet_output_1,
                                            nestnet_output_2,
                                            nestnet_output_3,
                                            nestnet_output_4,
                                            nestnet_output_denoise])
    else:
        model = Model(inputs=inputs, outputs=nestnet_output_denoise)

    return model



def model_DeepPDHMnet(inputs, n_labels, dropout=0, using_deep_supervision=False):
    """
    Model to create a Deep P-DHM Net.
    
    inputs: Input layer of the model.
    n_labels: Numbers of output layers requiered.
    dropout: Rate of dropout at each dropout layer.
    using_deep_supervision: Decouple all outputs if True.
    """
    
    # Size of each layers
    nb_filter = [64, 64, 64, 64, 64, 64]

    # Set image data format to channels first
    global bn_axis
    K.set_image_data_format("channels_last")
    bn_axis = -1
    
    ##### 1-1
    ## Input : [inputs]
    ## Outputs : [2-1, 1-2, 1-3, 1-4, 1-5, 1-6]
    ## Transform : (batch_size, a, b, c) ---> (batch_size, a/2, b/2, nb_filter[0])
    conv1_1 = conv_batchnorm_relu_block(inputs, nb_filter=nb_filter[0], name="conv2d_1-1")
    pool1_1 = MaxPooling2D((2, 2), name="maxpooling_1-1")(conv1_1)
    pool1_1 = Dropout(dropout, name="dropout_1-1")(pool1_1)
    
    ##### 2-1
    ## Input : [1-1]
    ## Outputs : [3-1, 2-2, 2-3, 2-4, 2-5, 1-2]
    ## Transform : (batch_size, a/2, b/2, nb_filter[0]) ---> (batch_size, a/4, b/4, nb_filter[1])
    conv2_1 = conv_batchnorm_relu_block(pool1_1, nb_filter=nb_filter[1], name="conv2d_2-1")
    pool2_1 = MaxPooling2D((2, 2), name="maxpooling_2-1")(conv2_1)
    pool2_1 = Dropout(dropout, name="dropout_2-1")(pool2_1)
    
    ##### 3-1
    ## Input : [2-1]
    ## Outputs : [4-1, 3-2, 3-3, 3-4, 2-2]
    ## Transform : (batch_size, a/4, b/4, nb_filter[1]) ---> (batch_size, a/8, b/8, nb_filter[2])
    conv3_1 = conv_batchnorm_relu_block(pool2_1, nb_filter=nb_filter[2], name="conv2d_3-1")
    pool3_1 = MaxPooling2D((2, 2), name="maxpooling_3-1")(conv3_1)
    pool3_1 = Dropout(dropout, name="dropout_3-1")(pool3_1)
    
    ##### 4-1
    ## Input : [3-1]
    ## Outputs : [5-1, 4-2, 4-3, 3-2]
    ## Transform : (batch_size, a/8, b/8, nb_filter[2]) ---> (batch_size, a/16, b/16, nb_filter[3])
    conv4_1 = conv_batchnorm_relu_block(pool3_1, nb_filter=nb_filter[3], name="conv2d_4-1")
    pool4_1 = MaxPooling2D((2, 2), name="maxpooling_4-1")(conv4_1)
    pool4_1 = Dropout(dropout, name="dropout_4-1")(pool4_1)
    
    ##### 5-1
    ## Input : [4-1]
    ## Outputs : [6-1, 5-2, 4-2]
    ## Transform : (batch_size, a/16, b/16, nb_filter[3]) ---> (batch_size, a/32, b/32, nb_filter[4])
    conv5_1 = conv_batchnorm_relu_block(pool4_1, nb_filter=nb_filter[4], name="conv2d_5-1")
    pool5_1 = MaxPooling2D((2, 2), name="maxpooling_5-1")(conv5_1)
    pool5_1 = Dropout(dropout, name="dropout_5-1")(pool5_1)
    
    ##### 6-1
    ## Input : [5-1]
    ## Outputs : [5-2]
    ## Transform : (batch_size, a/32, b/32, nb_filter[4]) ---> (batch_size, a/32, b/32, nb_filter[5])
    conv6_1 = conv_batchnorm_relu_block(pool5_1, nb_filter=nb_filter[5], name="conv2d_6-1")
    conv6_1 = Dropout(dropout, name="dropout_6-1")(conv6_1)
    
    ##### 1-2
    ## Input : [2-1, 1-1]
    ## Outputs : [1-3, 1-4, 1-5, 1-6, outputs]
    ## Transform : (batch_size, a, b, nb_filter[1]/4 + nb_filter[0]) ---> (batch_size, a, b, nb_filter[0])
    up1_2 = SubpixelConv2D(conv2_1,'up_1-2',scale=2)(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge_1-2', axis=bn_axis)
    conv1_2 = conv_batchnorm_relu_block(conv1_2,  nb_filter=nb_filter[0], name="conv2d_1-2")
    conv1_2 = Dropout(dropout, name="dropout_1-2")(conv1_2)
    
    ##### 2-2
    ## Input : [3-1, 2-1]
    ## Outputs : [2-3, 2-4, 2-5, 1-3]
    ## Transform : (batch_size, a/2, b/2, nb_filter[2]/4 + nb_filter[1]) ---> (batch_size, a/2, b/2, nb_filter[1])
    up2_2 = SubpixelConv2D(conv3_1,'up_2-2',scale=2)(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge_2-2', axis=bn_axis)
    conv2_2 = conv_batchnorm_relu_block(conv2_2,  nb_filter=nb_filter[1], name="conv2d_2-2")
    conv2_2 = Dropout(dropout, name="dropout_2-2")(conv2_2)
    
    ##### 3-2
    ## Input : [4-1, 3-1]
    ## Outputs : [3-3, 3-4, 2-3]
    ## Transform : (batch_size, a/4, b/4, nb_filter[3]/4 + nb_filter[2]) ---> (batch_size, a/4, b/4, nb_filter[2])
    up3_2 = SubpixelConv2D(conv4_1,'up_3-2',scale=2)(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge_3-2', axis=bn_axis)
    conv3_2 = conv_batchnorm_relu_block(conv3_2,  nb_filter=nb_filter[2], name="conv2d_3-2")
    conv3_2 = Dropout(dropout, name="dropout_3-2")(conv3_2)
    
    ##### 4-2
    ## Input : [5-1, 4-1]
    ## Outputs : [4-3, 3-3]
    ## Transform : (batch_size, a/8, b/8, nb_filter[4]/4 + nb_filter[3]) ---> (batch_size, a/8, b/8, nb_filter[3])
    up4_2 = SubpixelConv2D(conv5_1,'up_4-2',scale=2)(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge_4-2', axis=bn_axis)
    conv4_2 = conv_batchnorm_relu_block(conv4_2,  nb_filter=nb_filter[3], name="conv2d_4-2")
    conv4_2 = Dropout(dropout, name="dropout_4-2")(conv4_2)
    
    ##### 5-2
    ## Input : [6-1, 5-1]
    ## Outputs : [4-3]
    ## Transform : (batch_size, a/16, b/16, nb_filter[5]/4 + nb_filter[4]) ---> (batch_size, a/16, b/16, nb_filter[4])
    up5_2 = SubpixelConv2D(conv6_1,'up_5-2',scale=2)(conv6_1)
    conv5_2 = concatenate([up5_2, conv5_1], name='merge_5-2', axis=bn_axis)
    conv5_2 = conv_batchnorm_relu_block(conv5_2,  nb_filter=nb_filter[4], name="conv2d_5-2")
    conv5_2 = Dropout(dropout, name="dropout_5-2")(conv5_2)
    
    ##### 1-3
    ## Input : [2-2, 1-1, 1-2]
    ## Outputs : [1-4, 1-5, 1-6, outputs]
    ## Transform : (batch_size, a, b, nb_filter[1]/4 + nb_filter[0] + nb_filter[0]) ---> (batch_size, a, b, nb_filter[0])
    up1_3 = SubpixelConv2D(conv2_2,'up_1-3',scale=2)(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge_1-3', axis=bn_axis)
    conv1_3 = conv_batchnorm_relu_block(conv1_3,  nb_filter=nb_filter[0], name="conv2d_1-3")
    conv1_3 = Dropout(dropout, name="dropout_1-3")(conv1_3)
    
    ##### 2-3
    ## Input : [3-2, 2-1, 2-2]
    ## Outputs : [2-4, 2-5, 1-4]
    ## Transform : (batch_size, a/2, b/2, nb_filter[2]/4 + nb_filter[1] + nb_filter[1]) ---> (batch_size, a/2, b/2, nb_filter[1])
    up2_3 = SubpixelConv2D(conv3_2,'up_2-3',scale=2)(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge_2-3', axis=bn_axis)
    conv2_3 = conv_batchnorm_relu_block(conv2_3,  nb_filter=nb_filter[1], name="conv2d_2-3")
    conv2_3 = Dropout(dropout, name="dropout_2-3")(conv2_3)
    
    ##### 3-3
    ## Input : [4-2, 3-1, 3-2]
    ## Outputs : [3-4, 2-4]
    ## Transform : (batch_size, a/4, b/4, nb_filter[3]/4 + nb_filter[2] + nb_filter[2]) ---> (batch_size, a/4, b/4, nb_filter[2])
    up3_3 = SubpixelConv2D(conv4_2,'up_3-3',scale=2)(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge_3-3', axis=bn_axis)
    conv3_3 = conv_batchnorm_relu_block(conv3_3,  nb_filter=nb_filter[2], name="conv2d_3-3")
    conv3_3 = Dropout(dropout, name="dropout_3-3")(conv3_3)
    
    ##### 4-3
    ## Input : [5-2, 4-1, 4-2]
    ## Outputs : [3-4]
    ## Transform : (batch_size, a/8, b/8, nb_filter[4]/4 + nb_filter[3] + nb_filter[3]) ---> (batch_size, a/8, b/8, nb_filter[3])
    up4_3 = SubpixelConv2D(conv5_2,'up_4-3',scale=2)(conv5_2)
    conv4_3 = concatenate([up4_3, conv4_1, conv4_2], name='merge_4-3', axis=bn_axis)
    conv4_3 = conv_batchnorm_relu_block(conv4_3,  nb_filter=nb_filter[3], name="conv2d_4-3")
    conv4_3 = Dropout(dropout, name="dropout_4-3")(conv4_3)
    
    ##### 1-4
    ## Input : [2-3, 1-1, 1-2, 1-3]
    ## Outputs : [1-5, 1-6, outputs]
    ## Transform : (batch_size, a, b, nb_filter[1]/4 + nb_filter[0] + nb_filter[0] + nb_filter[0]) ---> (batch_size, a, b, nb_filter[0])
    up1_4 = SubpixelConv2D(conv2_3,'up_1-4',scale=2)(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge_1-4', axis=bn_axis)
    conv1_4 = conv_batchnorm_relu_block(conv1_4,  nb_filter=nb_filter[0], name="conv2d_1-4")
    conv1_4 = Dropout(dropout, name="dropout_1-4")(conv1_4)
    
    ##### 2-4
    ## Input : [3-3, 2-1, 2-2, 2-3]
    ## Outputs : [2-5, 1-5]
    ## Transform : (batch_size, a/2, b/2, nb_filter[2]/4 + nb_filter[1] + nb_filter[1] + nb_filter[1]) ---> (batch_size, a/2, b/2, nb_filter[1])
    up2_4 = SubpixelConv2D(conv3_3,'up_2-4',scale=2)(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge_2-4', axis=bn_axis)
    conv2_4 = conv_batchnorm_relu_block(conv2_4,  nb_filter=nb_filter[1], name="conv2d_2-4")
    conv2_4 = Dropout(dropout, name="dropout_2-4")(conv2_4)
    
    ##### 3-4
    ## Input : [4-3, 3-1, 3-2, 3-3]
    ## Outputs : [2-5]
    ## Transform : (batch_size, a/4, b/4, nb_filter[3]/4 + nb_filter[2] + nb_filter[2] + nb_filter[2]) ---> (batch_size, a/4, b/4, nb_filter[2])
    up3_4 = SubpixelConv2D(conv4_3,'up_3-4',scale=2)(conv4_3)
    conv3_4 = concatenate([up3_4, conv3_1, conv3_2, conv3_3], name='merge_3-4', axis=bn_axis)
    conv3_4 = conv_batchnorm_relu_block(conv3_4,  nb_filter=nb_filter[2], name="conv2d_3-4")
    conv3_4 = Dropout(dropout, name="dropout_3-4")(conv3_4)
    
    ##### 1-5
    ## Input : [2-4, 1-1, 1-2, 1-3, 1-4]
    ## Outputs : [1-6, outputs]
    ## Transform : (batch_size, a, b, nb_filter[1]/4 + nb_filter[0] + nb_filter[0] + nb_filter[0] + nb_filter[0]) ---> (batch_size, a, b, nb_filter[0])
    up1_5 = SubpixelConv2D(conv2_4,'up_1-5',scale=2)(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge_1-5', axis=bn_axis)
    conv1_5 = conv_batchnorm_relu_block(conv1_5,  nb_filter=nb_filter[0], name="conv2d_1-5")
    conv1_5 = Dropout(dropout, name="dropout_1-5")(conv1_5)
    
    ##### 2-5
    ## Input : [3-4, 2-1, 2-2, 2-3, 2-5]
    ## Outputs : [1-6]
    ## Transform : (batch_size, a/2, b/2, nb_filter[2]/4 + nb_filter[1] + nb_filter[1] + nb_filter[1] + nb_filter[1]) ---> (batch_size, a/2, b/2, nb_filter[1])
    up2_5 = SubpixelConv2D(conv3_4,'up_2-5',scale=2)(conv3_4)
    conv2_5 = concatenate([up2_5, conv2_1, conv2_2, conv2_3, conv2_4], name='merge_2-5', axis=bn_axis)
    conv2_5 = conv_batchnorm_relu_block(conv2_5,  nb_filter=nb_filter[1], name="conv2d_2-5")
    conv2_5 = Dropout(dropout, name="dropout_2-5")(conv2_5)
    
    ##### 1-6
    ## Input : [2-5, 1-1, 1-2, 1-3, 1-4, 1-5]
    ## Outputs : [outputs]
    ## Transform : (batch_size, a, b, nb_filter[1]/4 + nb_filter[0] + nb_filter[0] + nb_filter[0] + nb_filter[0] + nb_filter[0]) ---> (batch_size, a, b, nb_filter[0])
    up1_6 = SubpixelConv2D(conv2_5,'up_1-6',scale=2)(conv2_5)
    conv1_6 = concatenate([up1_6, conv1_1, conv1_2, conv1_3, conv1_4, conv1_5], name='merge_1-6', axis=bn_axis)
    conv1_6 = conv_batchnorm_relu_block(conv1_6,  nb_filter=nb_filter[0], name="conv2d_1-6")
    conv1_6 = Dropout(dropout, name="dropout_1-6")(conv1_6)    
    
    ##### outputs
    ## Inputs : [1-2, 1-3, 1-4, 1-5, 1-6]
    nestnet_output_1 = Conv2D(n_labels, (1, 1), activation='linear', name='output_1',padding='same')(conv1_2)
    nestnet_output_2 = Conv2D(n_labels, (1, 1), activation='linear', name='output_2', padding='same' )(conv1_3)
    nestnet_output_3 = Conv2D(n_labels, (1, 1), activation='linear', name='output_3', padding='same')(conv1_4)
    nestnet_output_4 = Conv2D(n_labels, (1, 1), activation='linear', name='output_4', padding='same')(conv1_5)
    nestnet_output_5 = Conv2D(n_labels, (1, 1), activation='linear', name='output_5', padding='same')(conv1_6)
    
    conv1_7 = concatenate([nestnet_output_1,nestnet_output_2, nestnet_output_3, nestnet_output_4, nestnet_output_5], name='merge_1-7', axis=bn_axis)
    nestnet_output_denoise = Conv2D(n_labels, (1, 1), activation='linear', name='output_denoised',padding='same')(conv1_7)
    
    if using_deep_supervision:
        model = Model(input=inputs, output=[nestnet_output_1,
                                            nestnet_output_2,
                                            nestnet_output_3,
                                            nestnet_output_4,
                                            nestnet_output_5,
                                            nestnet_output_denoise])
    else:
        model = Model(inputs=inputs, outputs=nestnet_output_denoise)

    return model



def model_unet(input_img, n_filters=8, layers_repetition=2, batchnorm=True, dropout=0, last_activation='linear'):
    '''
    Function to define a U-net architecture.
    Ref:
        [1] U-net: Convolutional Networks for Biomedical Image Segmentation.
        Ronneberger O., Fischer P. and Brox T.
        https://arxiv.org/abs/1505.04597

    Parameters:
    input_img: Input layer with the same size as the images.
    n_filters: Base number of filters, keep in mind that this number is double at each contraction.
    repetition: Number of repetition of the conv2d_block between each contraction and expansion.
    dropout: Rate at which the input is set to 0.
    batchnorm: If True, adds a BatchNormalization layers.

    Return:
    model: Keras model of a U-net.
    '''
    
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, repetition=layers_repetition,kernel_size=3,batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, repetition=layers_repetition,kernel_size=3,batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, repetition=layers_repetition,kernel_size=3,batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, repetition=layers_repetition,kernel_size=3,batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c6 = conv2d_block(p4, n_filters * 16, repetition=layers_repetition,kernel_size=3,batchnorm=batchnorm)
    
    # Expansive Path
    u7 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c4])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 8, repetition=layers_repetition,kernel_size=3,batchnorm=batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c3])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 4, repetition=layers_repetition,kernel_size=3,batchnorm=batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c2])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 2, repetition=layers_repetition,kernel_size=3,batchnorm=batchnorm)
    
    u10 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c9)
    u10 = concatenate([u10, c1])
    u10 = Dropout(dropout)(u10)
    c10 = conv2d_block(u10, n_filters * 1, repetition=layers_repetition,kernel_size=3,batchnorm=batchnorm)
    
    output_denoised = Conv2D(1, (1, 1), activation=last_activation, name='output_denoised')(c10)
    
    model = Model(inputs=[input_img], outputs=[output_denoised])
    
    return model