import cv2
import numpy as np
from skimage.transform import rescale

def apply_filter_to_image(image, filter):
    """
    This function applies a given filter to an input image using convolution.

    Parameters:
    image (numpy.ndarray): The input image to which the filter is to be applied. It should be a 3D array representing an RGB image.
    filter (numpy.ndarray): The filter to be applied to the image. It should be a 2D array.

    The function first separates the R, G, and B channels of the image. It then calculates the dimensions of the image and the filter, and computes the amount of padding needed for the convolution operation.

    The R, G, and B channels are then each padded using the 'reflect' mode, which reflects the values at the edge of the array.

    An output image is initialized as a zero array of the same shape as one of the color channels.

    The function then loops over each color channel, and for each pixel in the channel, it applies the filter by performing a convolution operation. The result is a new array for each color channel.

    These new arrays are then stacked together to form a 3D array representing the filtered image. The initial zero array is removed, and the filtered image is returned.

    Note: This function uses numpy functions such as np.pad, np.sum, np.asarray, np.reshape, and np.dstack.
    """
    # Separate the color channels
    red_channel, green_channel, blue_channel = cv2.split(image)
    
    # Calculate dimensions and padding
    image_height, image_width = red_channel.shape
    filter_height, filter_width = filter.shape
    height_padding, width_padding = int((filter_height-1)/2), int((filter_width-1)/2)
    
    # Define padding
    padding = ((height_padding, height_padding), (width_padding, width_padding))
    
    # Pad each color channel
    padded_channels = [np.pad(channel, pad_width=padding, mode='reflect') for channel in [red_channel, green_channel, blue_channel]]
    
    # Initialize output image
    output_image = np.zeros_like(red_channel)
    
    # Apply filter to each color channel
    for channel in padded_channels:
        filtered_channel = []
        
        # Perform convolution
        for row in range(image_height):
            for col in range(image_width):
                total = np.sum(np.multiply(channel[row:row+filter_height, col:col+filter_width], filter))
                filtered_channel.append(total)
                
        # Reshape and stack channels
        filtered_channel = np.asarray(filtered_channel).reshape(image_height, image_width)
        output_image = np.dstack((output_image, filtered_channel))
        
    # Remove initial zero array
    output_image = output_image[:, :, 1:]
    
    return output_image

def generate_gaussian_filter(shape=(3, 3), sigma=0.5):
    """
    This function generates a Gaussian filter with a given shape and standard deviation (sigma).

    Parameters:
    shape (tuple): The shape of the Gaussian filter. It should be a 2D tuple. Default is (3, 3).
    sigma (float): The standard deviation of the Gaussian filter. Default is 0.5.

    The function first calculates the center of the filter. It then generates a grid of (x, y) coordinates that represent the distance from the center.

    The Gaussian function is applied to each (x, y) coordinate to generate the filter. Any values in the filter that are close to zero (less than the machine epsilon times the maximum value in the filter) are set to zero.

    The filter is then normalized so that its sum is 1, and returned.

    Note: This function uses numpy functions such as np.ogrid, np.exp, np.finfo, and np.sum.
    """
    # Calculate the center of the filter
    center_x, center_y = [(dimension - 1) / 2. for dimension in shape]

    # Generate a grid of (x, y) coordinates
    y, x = np.ogrid[-center_x:center_x+1, -center_y:center_y+1]

    # Apply the Gaussian function to each (x, y) coordinate
    gaussian_filter = np.exp(-(x*x + y*y) / (2. * sigma * sigma))

    # Set values close to zero to zero
    gaussian_filter[gaussian_filter < np.finfo(gaussian_filter.dtype).eps * gaussian_filter.max()] = 0

    # Normalize the filter
    sum_of_values = gaussian_filter.sum()
    if sum_of_values != 0:
        gaussian_filter /= sum_of_values

    return gaussian_filter
