'''
Functions of the loss functions and metrics for denoising images
'''
import tensorflow as tf

def ssimsmape_loss(y_true, y_pred):
    '''
    Loss function meant for the denoising of images.
    
    Parameters:
    y_true: Tensor of the reference noiseless image.
    y_pred: Tensor of the prediction of the denoising of the noisy input.
    
    Return:
    Score of the loss function.
    '''
    return tf.reduce_mean(0.84*(1-tf.image.ssim(y_true, y_pred, 1.0))+
                          0.16*(abs(y_true - y_pred)/(abs(y_true)+abs(y_pred)+0.0001)))

def psnr(y_true, y_pred):
    '''
    Metric to evalute the peak signal to noise ratio between two images.
    
    Parameters:
    y_true: Tensor of the reference noiseless image.
    y_pred: Tensor of the prediction of the denoising of the noisy input.
    
    Return:
    Score of the metric.
    '''
    max_pixel = 1.0
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_pixel))

def ssim(y_true, y_pred):
    '''
    Metric to evalute the structural similarity index measure between two images.
    
    Parameters:
    y_true: Tensor of the reference noiseless image.
    y_pred: Tensor of the prediction of the denoising of the noisy input.
    
    Return:
    Score of the metric.
    '''
    max_pixel = 1.0
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_pixel))
