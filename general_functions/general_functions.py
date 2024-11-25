'''
Set of general functions
'''
import numpy as np

def robustnorm(img, percent=0.01):
    '''Robust Normalization of a batch of images.
    A certain percentage of the tails of the histogram are cut-off and set to 0 and 1 respectivelly.
    '''
    img = np.array(img)
    img_shape = img.shape
    min_img = np.tile(np.min(img, axis=(1,2))[:, np.newaxis, np.newaxis], (1,img_shape[1],img_shape[2]))
    max_img = np.tile(np.max(img, axis=(1,2))[:, np.newaxis, np.newaxis], (1,img_shape[1],img_shape[2]))
    
    prenorm_img = (img-((1-percent)*min_img)-(percent*max_img))/((max_img-min_img+1e-8)*(1-2*percent))
    tail1 = np.where(prenorm_img <= 0, 0, prenorm_img)
    return np.where(tail1 >= 1, 1, tail1)

def robustnorm_single(img, percent=0.01):
    '''Robust Normalization of a single image.
    A certain percentage of the tails of the histogram are cut-off and set to 0 and 1 respectivelly.
    '''
    img = np.array(img)
    img_shape = img.shape
    min_img = np.min(img)
    max_img = np.max(img)
    
    prenorm_img = (img-((1-percent)*min_img)-(percent*max_img))/((max_img-min_img+1e-8)*(1-2*percent))
    tail1 = np.where(prenorm_img <= 0, 0, prenorm_img)
    return np.where(tail1 >= 1, 1, tail1)

def validate_array(input_array, name):
    '''Validate that the input is an array of integers or floating-point numbers or just a regular
     integer or floating-point number.
    '''
    if input_array is None:
        return np.array([])
    if isinstance(input_array, (int, float)):
        return np.array([input_array])
    if (isinstance(input_array, (list, tuple, np.ndarray)) and
            all(isinstance(value, (int, float, np.int64, np.float64)) for value in input_array)):
        return np.array(input_array)
    raise TypeError(f"{name} must be an array of integers or floating-point numbers or an"
                    " integer or a floating-point number")

def format_uncertainty(value, uncertainty):
    '''Format the value and the uncertainty so that we respect the rules to present the results
    '''
    order = int(f"{uncertainty:.1e}"[-2:]) * (-1 if f"{uncertainty:.1e}"[-3] == "-" else 1)
    cropped_value = round(value / 10**(order+1), 2)
    cropped_uncertainty = round(uncertainty / 10**(order+1), 2)
    return cropped_value, cropped_uncertainty, order+1

def print_uncertainty(value, uncertainty, order):
    '''Print the uncertainty following the rules established.
    '''
    if int(order)==0:
        return fr"({value:.2f} $\pm$ {uncertainty})"
    return fr"({value:.2f} $\pm$ {uncertainty})$\times 10^{int(order):d}$"

#### Uncertainty patterns
# Change the NAME to the one of the variable considered
#        the UNIT to the one associated to the variable
#        the _VARNAME to the name of the associated internal variable

# @property
# def u_type_NAME(self):
#     '''Returns the type and value of the uncertainty for the NAME in UNIT.

#     Returns
#     -------
#     out : list
#         List of the type of uncertainty applied and its associated value.
#     '''
#     return self._VARNAME

# def set_u_type_NAME(self, value, fixed=True):
#     '''Defines the type and value of the uncertainty associated to the NAME in UNIT.

#     Parameters
#     ----------
#     value : int, float
#         Value of the uncertainty to apply.
#     fixed : bool
#         Defines if the uncertainty is fixed or a percentage of the measurement considered
#     '''
#     if fixed:
#         self._VARNAME = ["fixed", value]
#     else:
#         self._VARNAME = ["percentage", abs(value)]
