""" When you Fourier Transform an Image most of the fourier coefficients are quite small and
    can be truncated or thresholded away so we can zero out all of the small fourier coefficients
    retaining only the largest 1% or 2% of those FFT values and then

    When we inverse Fourier Transform that thresholded FFT signal we recover the original image
    with relatively low degradation

    So compression comes about by only saving these largest %1 or %2 fourier coefficients """

from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
import os

from mpl_toolkits.mplot3d import axes3d

path = r'C:\Users\oguzhan.sari\Desktop\Oğuzhan\development\fourier_SB\image compression FFT'
plt.rcParams['figure.figsize'] = [5, 5]
plt.rcParams.update({'font.size': 18})

A = imread(os.path.join(path, 'headshot.jpg'))
B = np.mean(A, -1)         # converts RGB to grayscale

plt.figure()
plt.imshow(A)
plt.axis('off')
plt.show()

plt.imshow(B)               # grayscale --> average of the color channels RGB
# plt.imshow(256 - A)       #,cmap='gray_r')         # steve-black background
plt.axis('off')
plt.show()
""" We just want to work on an array for simplicity, not to work on three color channel """


Bt = np.fft.fft2(B)
Btsorted = np.sort(np.abs(Bt.reshape(-1)))          # sort by magnitude

# Zero out all small coefficients and inverse transform
for keep in (0.1, 0.05, 0.01, 0.002):
    thresh = Btsorted[int(np.floor((1-keep)*len(Btsorted)))]
    ind = np.abs(Bt) > thresh                                   # find small indices
    Btlow = Bt * ind                                         # Threshold small indices  # mask [1, .. 0] ones and zeros
    Alow = np.fft.ifft2(Btlow).real                             # compressed image
    plt.figure()
    plt.imshow(Alow, cmap='gray')
    # plt.imshow(256- Alow, cmap='gray')
    plt.axis('off')
    plt.title('Compressed image: keep = ' + str(keep*100) + '%')
    plt.show()


""" There is a lot of high frequency content in hair, fur and grass, thats when we need to keep to reconstruct 
    that image and so high frequency stuff is harder to represent 

Pixar - Toy Story;
    Characters had plastic skin without hair and fur no grass 
    because it is much much easier to work in a compressed representation and to render those types of objects
    than to render hair, fur and grass 
    
    And it was big step from Toy Story to Monsters Inc where they finally figured out how to render big hairy monster
    because that's a lot harder, a lot higher frequency content, a lot more detail you have to render that you can't
    just compress out """

""" The Larger the image the more compressible it is
    So if we have an ultra ultra high-res image, there's more redundant information that we can compress out 
    We'll get a higher compression ratio the bigger the image
    
    If we have smaller image low-res, we need more of those fourier coefficients to reconstruct the image """

# Plot pixel intensity of the image as a surface

plt.rcParams['figure.figsize'] = [6, 6]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(np.arange(1, np.shape(B)[1] + 1), np.arange(1, np.shape(B)[0] + 1))
ax.plot_surface(X[0::10, 0::10], Y[0::10, 0::10], 256-B[0::10, 0::10], cmap='viridis', edgecolor='none')
ax.set_title = 'Surface Plot'
ax.mouse_init()
#ax.view_init(200, 270)
ax.view_init(-270, -270)              #different perspective
plt.show()

""" this is the exact headshot.jpg but rendered as a surface plot
    So brighter pixels in my face are both brighter and taller - kind of an elevation map
    
    My face kind of mountain landscape of pixel intensity 
    the whites of my eye were the brightest pixels, those are the tallest peaks
    
    What we are trying to do with fourier approximation is trying the approximate this mountain land scape
    of the pixel intensity as a sum of sin() and cosine() waves in X and in Y at different frequences 
So we are going to add up all of those kind of wavy sheets in just the right proportion and
we're going to reconstruct this topography of pixel intensities to reconstruct my face"""