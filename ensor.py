import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
#from ipywidgets import fixed, interact, widgets
import math
import warnings



### ================= ###
### General functions ###
### ================= ###


def plot_image(image, percentile = [0., 100.], ax = None, cmap = 'gray_r', figsize = (6,6), origin = 'lower', norm = None, plot_colorbar = False, title = ''):

    if ax is None:
        plt.figure(figsize = figsize)
    else:
        plt.sca(ax)

    vmin, vmax = np.nanpercentile(image, percentile[0]),np.nanpercentile(image, percentile[1])

    plt.imshow(image, origin = origin, cmap = cmap, vmin = vmin, vmax = vmax, norm = norm)
    plt.xticks([])
    plt.yticks([])

    if title != '':
        plt.title(title)

    if plot_colorbar:
        plt.colorbar()

    if ax is None:
        plt.show()
        


def plot_curve(func, plot_yaxis_log = False, plot_mag = True, axvlines = [], clip = True, ax = None, figsize=(7, 4), label = '', **kwargs):
    '''
    This only works for absolute scalings, not relative ones!
    '''
    xmin = 1e-5
    xmax = 15

    if len(axvlines) == 0:
        plot_axvlines = False
    else:
        plot_axvlines = True

    if plot_mag:# or plot_log
        x = np.logspace(np.log10(xmin) ,np.log10(xmax), num = 1_000)
    else:
        x = np.linspace(xmin,xmax, 1_000)

    y = func(x, **kwargs)
    ymin,ymax = np.min(y), np.max(y)

    if clip:
        y = np.clip(y,0,1)
        ymin,ymax = 1e-4,1

    if plot_mag:
        x = nanomaggy_to_mag(x)
        
    if plot_yaxis_log:
        #x = np.log10(x)
        y = np.log10(y)

        if clip:
            ymin = np.log10(ymin)
            ymax = np.log10(ymax)

    if ax is None:
        plt.figure(figsize = figsize)
    else:
        plt.sca(ax)
    plt.plot(x,y, lw=2, label = label)

    #plt.plot(x,y, lw=2)

    if plot_yaxis_log and not plot_mag:
        plt.xlabel("Nanomaggy")
        plt.ylabel("log(Output)")
    if not plot_yaxis_log and not plot_mag:
        plt.xlabel("Nanomaggy")
        plt.ylabel("Output")
    if plot_yaxis_log and plot_mag:
        plt.xlabel("AB mag")
        plt.ylabel("log(Output)")
    if not plot_yaxis_log and plot_mag:
        plt.xlabel("AB mag")
        plt.ylabel("Output")


    if plot_axvlines:
        #axvlines = [398, 0.25118, 0.015848] #in nanomaggy
        for axvline in axvlines:
            if not plot_mag:
                plt.axvline(axvline, c = 'k', ls ='--', zorder = -10)
            if plot_mag:
                plt.axvline(nanomaggy_to_mag(axvline), c = 'k', ls ='--', zorder = -10)
                
    
    plt.xlim(np.min(x), np.max(x))
    # plt.ylim(-0.1,2)
    plt.ylim(ymin, ymax)

    plt.grid(True, linestyle=":", alpha=0.6)

    if plot_mag:
        plt.gca().invert_xaxis()

    if ax is None:
        plt.show()


def estimate_noise(img):
    '''
    You can estimate the noise level empirically from an image by using the median absolute deviation
    '''

    return 1.4826 * np.nanmedian(np.abs(img - np.nanmedian(img)))


### ========================== ###
### Unit conversion functions ###
### ========================== ###


def njy_to_nanomaggy(img):
    """
    Nanomaggy is a linear unit. Equals 10^-9 (one-billionth) of a "maggie," which corresponds to a flux of an object with an AB magnitude of 22.5
    """
    return img/3631


def nanomaggy_to_njy(img):
    """
    Nanomaggy is a linear unit. Equals 10^-9 (one-billionth) of a "maggie," which corresponds to a flux of an object with an AB magnitude of 22.5
    """
    return img*3631




def nanomaggy_to_mag(flux_nmgy):
    """
    Convert flux in nanomaggies to AB magnitudes.
    Returns np.nan for flux <= 0.
    """
    flux = np.asanyarray(flux_nmgy)
    with np.errstate(invalid='ignore', divide='ignore'):
        mag = 22.5 - 2.5 * np.log10(flux)
        mag = np.where(flux > 0, mag, np.nan)
    return mag
    


def mag_to_nanomaggy(mag_ab):
    """
    Convert AB magnitude to nanomaggies

    Parameters
    ----------
    mag_ab : array_like or float
        AB magnitude.

    Returns
    -------
    flux_nmgy : ndarray or float
        Flux in nanomaggies.

    """
    mag = np.asanyarray(mag_ab, dtype=float)
    flux_nmgy = 10.0 ** (-0.4 * (mag - 22.5))
        
    return flux_nmgy




### ================= ###
### Scaling functions ###
### ================= ###


def linear(x, slope = 1, intercept = 0):
    '''
    f(x) = slope*x + intercept

    slope and intercept are both tuneable parameters.
    '''
    return slope*x+intercept

def log(x, a = 2):
    '''
    log(a*x + 1) / log(a)

    a is a tuneable parameter
    '''
    return np.log(a*x+1) / np.log(a+1)

def sqrt(x):
    '''
    f(x) = sqrt(x)

    No tuneable parameters.
    '''
    return np.sqrt(x)


def asinh_basic(x, beta = 10, x0 = 0):
    """
    f(x) = arcsinh((x-x0)/beta)

    beta is also known as softening parameter (or linear scale threshold).

    So if (x-x0) << beta it is linear,
    and if (x-x0) >> beta it is logarithmic.
    """
    return np.asinh((x-x0)/beta)


def asinh_alternative(x, Q = 10, stretch = 10, m = 0):
    """
    f(x) = arcsinh(Q * (x - m)/stretch) / Q,
    and clip at 0.

    This is a different parameterization of asinh.

    Q dictates where transition happens from linear to log.
    As Q -> 0, there is no log part. Just linear.
    As Q increases, log part stays longer

    Stretch is kind of like (inverse) gain. Smaller stretch helps faint features pop more, but also more noise.

    It helps to set stretch first, so as to tune the background, and then the Q, to determine where to switch to log.

    NOTE: some implementations do the /Q at the end differently. 
    """

    val = np.asinh(Q * (x - m) / stretch) / Q
    return np.maximum(val, 0)


def asinh(x, Q = 20, stretch = 1, m = 0):
    """
    This is the Legacy Survey / SDSS parameterization, which I like a lot
    https://github.com/legacysurvey/legacypipe/blob/f77ed93c4e9df40e27aec132fd9a9db0c78678d5/py/legacypipe/survey.py#L492
    
    f(i) = asinh(Q * I) / sqrt(Q). 
    Where I is img * stretch + m

    Q, stretch, and m are all tuneable parameters.

    Q dictates where transition happens from linear to log.
    As Q -> 0, there is no log part. Just linear.
    As Q increases, log part stays longer.

    Higher stretch helps faint features pop more, but also adds more noise. 
    
    Note that the denominator above is sqrt(Q), instead of Q in traditional asinh. This sqrt() helps boost faint signals even more.
    This is because, in traditional asinh, where:
    f(i) = asinh(Q*I)/sqrt(Q), in linear regime: f(i) = Q*I/Q = I. 

    Whereas now:
    f(i) = asinh(Q*I)/sqrt(Q), in linear regime: f(i) = Q*I/sqrt(Q) = I sqrt(Q). So boost upwards!
    
    """
   
    I = x * stretch + m
    val = np.asinh(Q * I) / np.sqrt(Q)
    
    return np.maximum(val, 0)



def mtf(x, m=0.5, x_max=100.0):
    """
    Midtones Transfer Function (MTF).

    It ensures that an input value of 0 maps to 0, the midpoint m maps exactly to 0.5, and 1 maps to 1.
    
    f(x) = ((m - 1) * x) / ((2 * m - 1) * x - m)

    Parameters
    ----------
    x : array_like or float
        Input data, in nanomaggies
    m : float, default 0.5
        Midtones balance parameter (0 < m < 1).
        - m = 0.5 results in a linear identity mapping (f(x) = x).
        - m < 0.5 brightens the midtones (pushes the curve upward).
        - m > 0.5 darkens the midtones (pulls the curve downward).
    x_max : float, default 100.0
            The maximum expected flux (or saturation point). used to 
            normalize the nanomaggies to [0, 1].

    Returns
    -------
    ndarray or float
        Transformed values clipped between 0 and 1.
    """


    # 1. Normalize nanomaggies to [0, 1]
    x_norm = np.clip(np.asanyarray(x, dtype=float) / x_max, 0.0, 1.0)
    
    # 2. Apply standard MTF formula
    with np.errstate(divide='ignore', invalid='ignore'):
        numerator = (m - 1.0) * x_norm
        denominator = (2.0 * m - 1.0) * x_norm - m
        
        val = np.where(np.isclose(m, 0.5), x_norm, numerator / denominator)
        
    return np.clip(val, 0.0, 1.0)



def mtf_inverse(x, alpha = 0.2, crop_fraction = 1, x_max = 100):
    """
    Instead of doing the standard MTF (see above) and providing an m, instead, we can calculate what the ideal m should be for a given image.
    This is done instead setting an alpha parameter, which is essentially the target brightness, and calculating what m should be to 
    ensure that the average brightness of the image is at alpha. 
    
    E.g., if alpha = 0.2, then calculate m to ensure the average brightness of this image to lands at 20% of maximum after applying MTF.

    The "average brightness" is calculated at a cutout of the image at the center, with a size of crop_fraction*size. 
    This typically works well for images with e.g., a central galaxy. 

    This is what we did for GZ JWST Cosmos-Web.

    This also implicitly assumes the image is a square...
    """
    assert crop_fraction >= 0 and crop_fraction <= 1, "crop_fraction should be between 0 and 1"

    # Normalize x by x_max
    x_norm = np.clip(np.asanyarray(x, dtype=float) / x_max, 0.0, 1.0)

    # Get average intensity x by zooming in on center of (assumed to be square) image.
    central_crop_size = int(crop_fraction * x_norm.shape[1])
    central_crop_low_edge = x_norm.shape[1]//2 - central_crop_size//2
    central_crop_high_edge = x_norm.shape[1]//2 + central_crop_size//2

    normalized_data_selected = x_norm[central_crop_low_edge:central_crop_high_edge, central_crop_low_edge:central_crop_high_edge]
    
    x_mean = np.mean(normalized_data_selected)

    target_m = (x_mean - alpha * x_mean) / (x_mean - 2 * alpha * x_mean + alpha)

    return mtf(x_norm, m=target_m, x_max=1.0)#We already normalized so x_max = 1

### ====================== ###
### Asinh helper functions ###
### ====================== ###

def calculate_Isat(Q):
    '''
    There is no maximum value, though when clip=True, we can calculate when I crosses 1. 
    The cool part is that this is uniquely decided by Q. Thus, when clipping between 0 and 1,
    all images are scaled the same way. 
    
    Derivation:

    f(I) = asinh(Q*I)/sqrt(Q). 

    Where this crosses one:

    1 = asinh(Q*Isat)/sqrt(Q) => asinh(Q*Isat) = sqrt(Q)

    Take sinh and rearrange:

    Isat = sinh(sqrt(Q))/Q.
    '''
    
    return np.sinh(np.sqrt(Q))/Q




def calculate_x_sat(Q, weight, stretch, m):
    '''
    There is no maximum value, though when clip=True, we can calculate when I crosses 1. 
    The cool part is that this is uniquely decided by the parameters. Thus, when clipping between 0 and 1,
    all images are scaled the same way. 
    
    Derivation:

    f(I) = asinh(Q*I)/sqrt(Q). 

    Where this crosses one:

    1 = asinh(Q*Isat)/sqrt(Q) => asinh(Q*Isat) = sqrt(Q)

    Take sinh and rearrange:

    Isat = sinh(sqrt(Q))/Q.
    

    we also know: I = x * weights * stretch + m, thus:

    x_sat = (Isat - m) / (weights * stretch)
    
    
    '''
    Isat = np.sinh(np.sqrt(Q))/Q
    xsat = (Isat - m) / (weight * stretch)
    return xsat



def calculate_x_zero(Q, weight, stretch, m):
    '''
    Derivation:

    f(I) = asinh(Q*I)/sqrt(Q). 

    Where this crosses 0:

    0 = asinh(Q*I_0)/sqrt(Q) => asinh(Q*I0) = 0

    Take sinh and rearrange:

    Q * I_0 = 0 => I0 = 0

    we also know: I = x * weight * stretch + m, thus:

    x_0 * weight * stretch + m = 0

    => x_0 = -m / (weight * stretch)
    '''
    return -m / (weight * stretch)


    


def calculate_x_knee(Q, weight, stretch, m):
    """
    asinh goes from linear to log. The "knee" is where it switched from linear to log.
    You can calculate the knee as follows. The knee is defined at where asinh(u) = Q*I = 1. So:

    u = Q * I = 1. So:

    I_knee = 1/Q. 

    So, Q uniquely defines where the knee happens in terms of I. But where does this correspond to in terms of the x-axis?
    
    We know: I = x * weights * stretch + m, thus:

    x_knee * weights * stretch + m = 1/Q

    => x_knee = (1/Q - m) / (weights * stretch)
    

    NOTE: if weight for a band a complicated dict, just add all the weights together. 
    """
    
    return (1/Q - m) / (weight*stretch)







### ============== ###
### Main functions ###
### ============== ###


# def make_image_old(imgs, bands, weights = None, func = None, clip=True, **func_kwargs):
    
#     if weights is None:
#         raise Exception('Need to input weights')

#     if func is None:
#         raise Exception('Need to input func')


#     img_dict = {}
#     for img, band_name in zip(imgs, bands):
#         img_dict[band_name] = img

#     red = sum([img_dict[band_name] * weights[band_name][0] for band_name in bands])
#     green = sum([img_dict[band_name] * weights[band_name][1] for band_name in bands])
#     blue = sum([img_dict[band_name] * weights[band_name][2] for band_name in bands])

#     red = func(red, **func_kwargs)
#     green = func(green, **func_kwargs)
#     blue = func(blue, **func_kwargs)

    
#     rgb = np.stack([red, green, blue], axis=-1)
#     if clip:
#         rgb = np.clip(rgb,0,1) 
#     return rgb





def make_image(imgs, bands, weights = None, func = None, clip=True, offset = 0.0, eps = 1e-10, apply_lupton = True, **func_kwargs):
    '''
    DESCRIPTION
    -----------
    A (hopefully) easy-to-use function to create greyscale or colour (rgb) images from astronomical observations. 

    The input is assumed to be in nanomaggies (https://www.sdss3.org/dr8/algorithms/magnitudes.php), but will work in principle with any linear flux unit.

    This is based on Lupton+2004 (https://arxiv.org/abs/astro-ph/0312483). Also inspired by the Legacy/SDSS method (https://github.com/legacysurvey/legacypipe/blob/f77ed93c4e9df40e27aec132fd9a9db0c78678d5/py/legacypipe/survey.py#L719) and Astropy's make_lupon_rgb (https://docs.astropy.org/en/stable/api/astropy.visualization.make_lupton_rgb.html).
    
    

    PARAMETERS
    ----------
    imgs : list
        A list of the astronomical observations in different bands, e.g., [g,r,i].

    bands : list
        The names of the bands of the observations in `imgs'. Will be used to connect the images to their corresponding weights.

    weights : dict
        A dictionary that maps all bands specified in `bands' with their corresponding weight in the different RGB channels. For each band, there should be a tuple with 3 elements, representing the weight in red, green, and blue channels. See notes below for a few examples.

    func : None
        The scaling (or stretch) function that maps flux values to pixel values. This can be the built-in functions (e.g., arcsinh, linear, log), or a custom defined function.

    clip : bool, default True
        Whether to, after applying the scaling/stretch function, clip the image between [0,1]. 

    offset : float, default 0.0
        This will be added to all channels prior to applying the scaling/stretch function. This can help with dealing with the random noise.
    
    eps : float, default 1e-10.
        To avoid division by 0. Should represent a negligible amount of flux. If input units are in nanomaggies, then the default value of  1e-10 corresponds to 47.5 AB mag. 

    apply_lupton : bool, default True
        Whether to combine the bands into an rgb according to the scheme described in Lupton+2004. Strongly advised to keep this to True, as this setting ensures that the colors are retained, even in saturated objects. In general, images will look more colorful when apply_lupton = True.

    **func_kwargs
        Any other arguments will be passed straight to `func'.
        

    RETURNS
    -------
    A MxNx3 rgb image with pixel values between 0 and 1 (if clip = True), where the original input dimensions were MxN.



    NOTES 
    -----

    --- Example of different weights dictionaries
    
    Let's image you have 3 bands: g,r, and i. The most basic weight dictionary maps the g-band to the blue channel, r-band to the green channel, and the i-band to the red channel:
    ```  
        weights = dict(
        g = (0, 0, 1), 
        r = (0, 1, 0),
        i = (1, 0, 0),
        )
    ```

    Below is an example of the Legacy/SDSS weighting, where the g-band was upweighted relative to the r- and i-bands:
    ```
    weights = dict(
        g = (0, 0, 6.0),
        r = (0, 3.4, 0.),
        i = (3.0,0, 0),
        )
    ```

    Image you have four bands, e.g., F115W, F150W, F200W, F227W, and F444W. You could combine them like the example below. Here, F115W and F150W are combined in the blue channel, F200W and F227W are combined to the green channel, and F444W is put in the red channel.
    ```
    weights = dict(
        F115W = (0, 0, 0.3),
        F150W = (0, 0, 0.7),
        F200W = (0, 0.5, 0),
        F227W = (0, 0.5, 0),
        F444W = (1, 0, 0),
        )
    ```



    EXAMPLE
    -------

    imgs = [g,r,i] #pre-loaded MxN arrays in nanomaggies for the g, r, and i bands
    bands = ['g','r','i']
    
    weights = dict(
        g = (0, 0, 1), 
        r = (0, 1, 0),
        i = (1, 0, 0),
    )
    
    rgb = make_image(imgs, bands, weights, func = asinh)
        
    '''

    # Setup
    if weights is None:
        weights = {}
        for band in bands:
            weights[band] = [1]*3
        #raise Warning('You did not input any weights. Creating a greyscale image with all bands.')
        warnings.warn('You did not input any weights. Creating a greyscale image with all bands.', UserWarning)
        

    if func is None:
        raise Exception("You need to input a scaling/stretch function. Recommended to set 'func = asinh'. ")

    img_dict = {}
    for img, band_name in zip(imgs, bands):
        img_dict[band_name] = img

        
    # Sum bands into rgb channels according to weight and offset
    r = sum([img_dict[band_name] * weights[band_name][0] for band_name in bands]) + offset
    g = sum([img_dict[band_name] * weights[band_name][1] for band_name in bands]) + offset
    b = sum([img_dict[band_name] * weights[band_name][2] for band_name in bands]) + offset

    # Remove negative values
    r = np.maximum(r, 0)
    g = np.maximum(g, 0)
    b = np.maximum(b, 0)


    if apply_lupton:
        # Calculate I
        I = (r+g+b)/3
    
        # Apply scaling/stretch function to I
        fI = func(I, **func_kwargs)
        
        # Scale individual channels 
        I += (I == 0.) * eps
        R = r/I * fI
        G = g/I * fI
        B = b/I * fI

    else: 
        R = func(r, **func_kwargs)
        G = func(g, **func_kwargs)
        B = func(b, **func_kwargs)
        
        
    # Stack
    rgb = np.stack([R, G, B], axis=-1)

    # Clip
    if clip:
        rgb = np.clip(rgb,0,1) 
        
    return rgb
