import math

import numpy as np
from PIL import Image
from skimage import color, io


def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    

    ### YOUR CODE HERE
    # Use skimage io.imread
    out = io.imread(image_path)
    ### END YOUR CODE

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255
    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    

    ### YOUR CODE HERE
    out = image * image /2
    ### END YOUR CODE

    return out


def convert_to_grey_scale(image):
    """Change image to gray scale.

    HINT: Look at `skimage.color` library to see if there is a function
    there you can use.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width).
    """
   

    ### YOUR CODE HERE
    out = color.rgb2gray(image)
    ### END YOUR CODE

    return out


def rgb_exclusion(image, channel):
    """Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "R", "G" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
   

    ### YOUR CODE HERE
    out = image.copy()
    if channel == 'R':
        out[:, :, 0] = 0
    elif channel == 'G':
        out[: , : , 1] = 0
    else:
        out[: , :, 2] = 0
        
        
    
    
    ### END YOUR CODE

    return out


def lab_decomposition(image, channel):
    """Decomposes the image into LAB and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "L", "A" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    lab = color.rgb2lab(image)
   

    ### YOUR CODE HERE
    out = lab.copy()
    if channel == 'L':
        out = out[:,:,0]
    elif channel == 'A':
        out = out[: , : , 1] 
    else:
        out = out[: , :, 2]
    ### END YOUR CODE

    return out


def hsv_decomposition(image, channel='H'):
    """Decomposes the image into HSV and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "H", "S" or "V".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    hsv = color.rgb2hsv(image)
    

    ### YOUR CODE HERE
    out = hsv.copy()
    if channel == 'H':
        out = out[:,:,0]
    elif channel == 'S':
        out = out[: , : , 1] 
    else:
        out = out[: , :, 2]
    ### END YOUR CODE

    return out


def mix_images(image1, image2, channel1, channel2):
    """Combines image1 and image2 by taking the left half of image1
    and the right half of image2. The final combination also excludes
    channel1 from image1 and channel2 from image2 for each image.

    HINTS: Use `rgb_exclusion()` you implemented earlier as a helper
    function. Also look up `np.concatenate()` to help you combine images.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).
        image2: numpy array of shape(image_height, image_width, 3).
        channel1: str specifying channel used for image1.
        channel2: str specifying channel used for image2.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    
    ### YOUR CODE HERE
    ###get out as image1 without chanel1, so is img2
    out = rgb_exclusion(image1, channel1)
    img2 = rgb_exclusion(image2,channel2)
    ### to get row and column
    img_tmp = image1.copy()
    img_tmp = img_tmp[:, : ,0]
    row = np.size(img_tmp, 0)
    col = np.size(img_tmp, 1)
    length1 = col // 2
    ### switch the right of img1
    out[:, length1 : col, :] = img2[:, length1 : col, :]
    ### END YOUR CODE

    return out


def mix_quadrants(image):
    """THIS IS AN EXTRA CREDIT FUNCTION.

    This function takes an image, and performs a different operation
    to each of the 4 quadrants of the image. Then it combines the 4
    quadrants back together.

    Here are the 4 operations you should perform on the 4 quadrants:
        Top left quadrant: Remove the 'R' channel using `rgb_exclusion()`.
        Top right quadrant: Dim the quadrant using `dim_image()`.
        Bottom left quadrant: Brighthen the quadrant using the function:
            x_n = x_p^0.5
        Bottom right quadrant: Remove the 'R' channel using `rgb_exclusion()`.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    
    ### YOUR CODE HERE
    img1 = rgb_exclusion(image, 'R')
    img2 = dim_image(image)
    img3 = np.sqrt(image)
    img4 = rgb_exclusion(image, 'R')
    img_tmp = image[: , : , 0]
    row = np.size(img_tmp, 0)
    col = np.size(img_tmp, 1)
    l1 = row // 2
    l2 = col // 2
    out = img1.copy()
    out[0: l1,l2 : col, :] = img2[0: l1,l2 : col, :]
    out[l1: row, 0: l2, :] = img3[l1: row, 0: l2, :]
    out[l1: row, l2: col, :] = img4[l1: row, l2: col, :]
    ### END YOUR CODE

    return out
