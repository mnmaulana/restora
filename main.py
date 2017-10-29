import numpy as np
import numpy.random as npr
import fitsio as fts
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, convolve, convolve2d

def richardson_lucy(image, psf, iterations=50, clip=True):
    """Richardson-Lucy deconvolution.

    Parameters
    ----------
    image : ndarray
       Input degraded image (can be N dimensional).
    psf : ndarray
       The point spread function.
    iterations : int
       Number of iterations. This parameter plays the role of
       regularisation.
    clip : boolean, optional
       True by default. If true, pixel value of the result above 1 or
       under -1 are thresholded for skimage pipeline compatibility.

    Returns
    -------
    im_deconv : ndarray
       The deconvolved image.

    Examples
    --------
    >>> from skimage import color, data, restoration
    >>> camera = color.rgb2gray(data.camera())
    >>> from scipy.signal import convolve2d
    >>> psf = np.ones((5, 5)) / 25
    >>> camera = convolve2d(camera, psf, 'same')
    >>> camera += 0.1 * camera.std() * np.random.standard_normal(camera.shape)
    >>> deconvolved = restoration.richardson_lucy(camera, psf, 5)

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution
    """
    # compute the times for direct convolution and the fft method. The fft is of
    # complexity O(N log(N)) for each dimension and the direct method does
    # straight arithmetic (and is O(n*k) to add n elements k times)
    direct_time = np.prod(image.shape + psf.shape)
    fft_time =  np.sum([n*np.log(n) for n in image.shape + psf.shape])

    # see whether the fourier transform convolution method or the direct
    # convolution method is faster (discussed in scikit-image PR #1792)
    time_ratio = 40.032 * fft_time / direct_time

    if time_ratio <= 1 or len(image.shape) > 2:
        convolve_method = fftconvolve
    else:
        convolve_method = convolve

    image = image.astype(np.float)
    psf = psf.astype(np.float)
    im_deconv = 0.5 * np.ones(image.shape)
    psf_mirror = psf[::-1, ::-1]

    for _ in range(iterations):
        relative_blur = image / convolve_method(im_deconv, psf, 'same')
        im_deconv *= convolve_method(relative_blur, psf_mirror, 'same')

    if clip:
        im_deconv[im_deconv > 1] = 1
        im_deconv[im_deconv < -1] = -1

    return im_deconv

def psf_gauss(A,k,l,sigma):
    kl = (k-x)**2 + (l-y)**2
    f_kl = A*np.exp(-kl/(2*sigma**2))
    return (f_kl)

image = np.array(fts.read('BLUR4.FIT'))
image += 1

#A = 10.0
k,l = np.meshgrid(np.arange(10),np.meshgrid(10))
#sigma = 5.0

x = np.median(k)
y = np.median(l)


psf = psf_gauss(100,k,l,5)


#blur = convolve(image,psf,'same')
#deconv10 = richardson_lucy(image,psf, 10, clip=False)  
deconv30 = richardson_lucy(image,psf, 30, clip=False)
deconv50 = richardson_lucy(image,psf, 50, clip=False)

#plt.imshow(image)
#plt.subplot(121)
#plt.imshow(image, cmap='gray')
#plt.contour(image)
#plt.title('Citra terdegradasi')
#plt.subplot(132)
#plt.imshow(blur)
#plt.subplot(122)
#plt.imshow(deconv10, cmap="gray")
#plt.contour(deconv)
#plt.title('Restorasi RL i=10')
plt.subplot(121)
plt.imshow(deconv30, cmap="gray")
#plt.contour(deconv)
plt.title('Restorasi RL i=30')
plt.subplot(122)
plt.imshow(deconv50, cmap="gray")
#plt.contour(deconv)
plt.title('Restorasi RL i=50')

plt.show()
"""
v0 = []
v1 = []

for i in range(len(image)):
    v0.append(image[i][50])
    v1.append(deconv[i][50])

plt.subplot(211)
plt.plot(range(len(image)),v0)
plt.subplot(212)
plt.plot(range(len(image)),v1)
plt.show
"""

"""
image = np.ones((100,100))

image[18][41] = 80
image[16][43] = 150
image[75][45] = 100
image[50][60] = 250

psf = np.ones((5, 5)) * 5
blurred = convolve2d(image, psf, 'same')

x1 = richardson_lucy(blurred, psf, 1, clip=False)
x5 = richardson_lucy(blurred, psf, 5, clip=False)
x10 = richardson_lucy(blurred, psf, 10, clip=False)
x20 = richardson_lucy(blurred, psf, 20, clip=False)

#fftcon = image / convolve(image,psf, 'same')
#im_deconv = convolve(fftcon, psf_mirror, 'same')
#conv = convolve(image,psf, 'same')


plt.figure(1)
plt.subplot(231)
plt.imshow(image,cmap='gray')
plt.title('Citra Sebenarnya')

plt.subplot(232)
plt.imshow(blurred,cmap='gray')
plt.title('Citra Terdegradasi (*psf)')

plt.subplot(233)
plt.imshow(x1,cmap='gray')
plt.title('Restorasi RL i=1')

plt.subplot(234)
plt.imshow(x5,cmap='gray')
plt.title('Restorasi RL i=5')

plt.subplot(235)
plt.imshow(x10,cmap='gray')
plt.title('Restorasi RL i=10')

plt.subplot(236)
plt.imshow(x20,cmap='gray')
plt.title('Restorasi RL i=20')
plt.show()

#print(im_deconv
"""

'''
img = mpimg.imread('sample.png')
psf = np.ones((5, 5)) / 25

blurred = convolve(img,psf, 'same')

plt.figure(1)
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(blurred)
plt.show()
'''
