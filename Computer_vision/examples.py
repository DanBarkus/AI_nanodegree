# Image color filtering (blue screen)

import matplotlib.pyplot as plt
import numpy as np
import cv2

%matplotlib inline

# Read in the image
image = cv2.imread('images/pizza_bluescreen.jpg')

# Print out the type of image data and its dimensions (height, width, and color)
print('This image is:', type(image), 
      ' with dimensions:', image.shape)

# Make a copy of the image
image_copy = np.copy(image)

# Change color to RGB (from BGR)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

# Display the image copy
plt.imshow(image_copy)

# play around with these values until you isolate the blue background
lower_blue = np.array([0,0,200]) 
upper_blue = np.array([250,250,255])

# Define the masked area
mask = cv2.inRange(image_copy, lower_blue, upper_blue)

# Vizualize the mask
plt.imshow(mask, cmap='gray')

# Mask the image to let the pizza show through
masked_image = np.copy(image_copy)

masked_image[mask != 0] = [0, 0, 0]

# Display it!
plt.imshow(masked_image)

# Load in a background image, and convert it to RGB 
background_image = cv2.imread('images/space_background.jpg')
background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)

# Crop it to the right size (514x816)
crop_background = background_image[0:514, 0:816]
# or
crop_background = cv2.resize(background_image, dsize=(812,514))

# Mask the cropped background so that the pizza area is blocked
crop_background[mask == 0] = [0, 0, 0]

# Display the background
plt.imshow(crop_background)

# Add the two images together to create a complete image!
complete_image = masked_image + crop_background

# Display the result
plt.imshow(complete_image)



# Useful Functions
# --------------------------------------------------------------

def standardize(image_list):
    
    # Empty image data array
    standard_list = []

    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardize_input(image)

        # Create a numerical label
        binary_label = encode(label)    

        # Append the image, and it's one hot encoded label to the full, processed list of image data 
        standard_list.append((standardized_im, binary_label))
        
    return standard_list

# Standardize all training images
STANDARDIZED_LIST = standardize(IMAGE_LIST)



# Sobel Filter
# Detects vertical lines
np.array([[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]])

filtered_image = cv2.filter2d(image, -1, filter)

# create binary image
cv2.threshold()

# Gaussian Blur
cv2.GaussianBlur(image, (5,5), 0)

# Canny Edge Detection
cv2.Canny()

# Hough transform
rho = 1                 # Resolution settings
theta = np.pi/180       # Resolution settings
threshold = 60          # Min Hough space intersections to detect line
min_line_length = 50    # Min length to be considered line (edge detected input image)
max_line_gap = 5        # Max gaps between lines (edge detected input image)

cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
# Returns 2 x,y coordinates representing line

# Hough Circles
minDist = 45            # Minimum distance between circles
param1 = 70             # Higher value for Canny edge detection
param2 = 11             # Threshold for circle detection (smaller value = fewer circles)
minRadius = 20
maxRadius = 40

cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, minDist, param1, param2, minRadius, maxRadius)
# 