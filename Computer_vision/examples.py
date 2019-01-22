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


# Harris Corner Detection
    # Convert to grayscale
    gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
    gray = np.float32(gray)

    # Detect corners 
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # Dilate corner image to enhance corner points
    dst = cv2.dilate(dst,None)

    plt.imshow(dst, cmap='gray')

# Dialation and Erosion
    # Reads in a binary image
    image = cv2.imread(‘j.png’, 0) 

    # Create a 5x5 kernel of ones
    kernel = np.ones((5,5),np.uint8)

    # Dilate the image
    dilation = cv2.dilate(image, kernel, iterations = 1)

    # Erode the image
    erosion = cv2.erode(image, kernel, iterations = 1)

    # Opening - erodes then dilates to get rid of noise
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # Closing - dilates then erodes to close holes in boundaries
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# Contor Detection
    # Convert to grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    # Create a binary thresholded image
    retval, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)

    # Find contours from thresholded, binary image
    retval, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw all contours on a copy of the original image
    contours_image = np.copy(image)
    contours_image = cv2.drawContours(contours_image, contours, -1, (0,255,0), 3)

    # Fit an ellipse to a contour and extract the angle from that ellipse
    (x,y), (MA,ma), angle = cv2.fitEllipse(selected_contour)  


# Bounding Box
    # Find the bounding rectangle of a selected contour
    x,y,w,h = cv2.boundingRect(selected_contour)

    # Draw the bounding rectangle as a purple box
    box_image = cv2.rectangle(contours_image, (x,y), (x+w,y+h), (200,0,200),2)

    # Crop using the dimensions of the bounding rectangle (x, y, w, h)
    cropped_image = image[y: y + h, x: x + w] 

# K-Means Clustering
    # Reshape image into a 2D array of pixels and 3 color values (RGB)
    pixel_vals = image.reshape((-1,3))

    # Convert to float type
    pixel_vals = np.float32(pixel_vals)

    # define stopping criteria
    # you can change the number of max iterations for faster convergence!
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    ## TODO: Select a value for k
    # then perform k-means clustering
    k = 3
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert data into 8-bit values
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]

    # reshape data into the original image dimensions
    segmented_image = segmented_data.reshape((image.shape))
    labels_reshape = labels.reshape(image.shape[0], image.shape[1])

# Image Pyramids
    level_1 = cv2.pyrDown(image)
    level_2 = cv2.pyrDown(level_1)
    level_3 = cv2.pyrDown(level_2) 

# ORB
    cv2.ORB_create(nfeatures = 500,         # Number of features to locate
                scaleFactor = 1.2,          # Pyramid decimation ratio (must be > 1)
                nlevels = 8,                # Number of pyramid levels
                edgeThreshold = 31,         # Size of border where features are not detected
                firstLevel = 0,             # Which level is the first in the pyramid
                WTA_K = 2,                  # Number of random pixels chosen during BRIEF (2,3,4)
                scoreType = HARRIS_SCORE,   # HARRIS_SCORE or FAST_SCORE, Fast is faster, Harris is better
                patchSize = 31,             # Size of patch used for BRIEF
                fastThreshold = 20)
    # nFeatures and scale are usually all you need to change


    # Set the parameters of the ORB algorithm by specifying the maximum number of keypoints to locate and
    # the pyramid decimation ratio
    orb = cv2.ORB_create(200, 2.0)

    # Find the keypoints in the gray scale training image and compute their ORB descriptor.
    # The None parameter is needed to indicate that we are not using a mask.
    keypoints, descriptor = orb.detectAndCompute(training_gray, None)

    # Create copies of the training image to draw our keypoints on
    keyp_without_size = copy.copy(training_image)
    keyp_with_size = copy.copy(training_image)

    # Draw the keypoints without size or orientation on one copy of the training image 
    cv2.drawKeypoints(training_image, keypoints, keyp_without_size, color = (0, 255, 0))

    # Draw the keypoints with size and orientation on the other copy of the training image
    cv2.drawKeypoints(training_image, keypoints, keyp_with_size, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Brute-Force matching
    cv2.BFMatcher(normType = cv2.NORM_L2,       # Metric used to determine quality of match (NORM_L2 - distance between 2 descriptors, NORM_HAMMING - counts dissimilar bits [for binary])
               crossCheck = false)              # TRUE performs matching twice, second time matches query images is matched to training image (instead of vice-versa)

    # Find the keypoints in the gray scale training and query images and compute their ORB descriptor.
    # The None parameter is needed to indicate that we are not using a mask in either case.
    keypoints_train, descriptors_train = orb.detectAndCompute(training_gray, None)
    keypoints_query, descriptors_query = orb.detectAndCompute(query_gray, None)

    # Create a Brute Force Matcher object. Set crossCheck to True so that the BFMatcher will only return consistent
    # pairs. Such technique usually produces best results with minimal number of outliers when there are enough matches.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

    # Perform the matching between the ORB descriptors of the training image and the query image
    matches = bf.match(descriptors_train, descriptors_query)

    # The matches with shorter distance are the ones we want. So, we sort the matches according to distance
    matches = sorted(matches, key = lambda x : x.distance)

    # Connect the keypoints in the training image with their best matching keypoints in the query image.
    # The best matches correspond to the first elements in the sorted matches list, since they are the ones
    # with the shorter distance. We draw the first 300 mathces and use flags = 2 to plot the matching keypoints
    # without size or orientation.
    result = cv2.drawMatches(training_gray, keypoints_train, query_gray, keypoints_query, matches[:300], query_gray, flags = 2)

# HOG
    cv2.HOGDescriptor(win_size = (64, 128),         # size of detection window (multiple of cell size)
                    block_size = (16, 16),          # size of block in pixels (multiple of cell size, smaller is finer detail)
                    block_stride = (8, 8),          # distance between adjacent blocks (bigger numbers are quicker but less accurate)
                    cell_size = (8, 8),             # size of cell in pixels (smaller is finer detail)
                    nbins = 9,                      # number of bins to use for histograms
                    win_sigma = DEFAULT_WIN_SIGMA,  # gaussian smoothing parameter
                    threshold_L2hys = 0.2,          # normalization method
                    gamma_correction = true,        # do or don't do gamma correction (improves performance)
                    nlevels = DEFAULT_NLEVELS)      # max detection window increases

    
    # Cell Size in pixels (width, height). Must be smaller than the size of the detection window
    # and must be chosen so that the resulting Block Size is smaller than the detection window.
    cell_size = (6, 6)

    # Number of cells per block in each direction (x, y). Must be chosen so that the resulting
    # Block Size is smaller than the detection window
    num_cells_per_block = (2, 2)

    # Block Size in pixels (width, height). Must be an integer multiple of Cell Size.
    # The Block Size must be smaller than the detection window
    block_size = (num_cells_per_block[0] * cell_size[0],
                num_cells_per_block[1] * cell_size[1])

    # Calculate the number of cells that fit in our image in the x and y directions
    x_cells = gray_image.shape[1] // cell_size[0]
    y_cells = gray_image.shape[0] // cell_size[1]

    # Horizontal distance between blocks in units of Cell Size. Must be an integer and it must
    # be set such that (x_cells - num_cells_per_block[0]) / h_stride = integer.
    h_stride = 1

    # Vertical distance between blocks in units of Cell Size. Must be an integer and it must
    # be set such that (y_cells - num_cells_per_block[1]) / v_stride = integer.
    v_stride = 1

    # Block Stride in pixels (horizantal, vertical). Must be an integer multiple of Cell Size
    block_stride = (cell_size[0] * h_stride, cell_size[1] * v_stride)

    # Number of gradient orientation bins
    num_bins = 9        

    # Specify the size of the detection window (Region of Interest) in pixels (width, height).
    # It must be an integer multiple of Cell Size and it must cover the entire image. Because
    # the detection window must be an integer multiple of cell size, depending on the size of
    # your cells, the resulting detection window might be slightly smaller than the image.
    # This is perfectly ok.
    win_size = (x_cells * cell_size[0] , y_cells * cell_size[1])

    # Set the parameters of the HOG descriptor using the variables defined above
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)

    # Compute the HOG Descriptor for the gray scale image
    hog_descriptor = hog.compute(gray_image)

            # Calculate the total number of blocks along the width of the detection window
            tot_bx = np.uint32(((x_cells - num_cells_per_block[0]) / h_stride) + 1)

            # Calculate the total number of blocks along the height of the detection window
            tot_by = np.uint32(((y_cells - num_cells_per_block[1]) / v_stride) + 1)

            # Calculate the total number of elements in the feature vector
            tot_els = (tot_bx) * (tot_by) * num_cells_per_block[0] * num_cells_per_block[1] * num_bins

            # Print the total number of elements the HOG feature vector should have
            print('\nThe total number of elements in the HOG Feature Vector should be: ',
                tot_bx, 'x',
                tot_by, 'x',
                num_cells_per_block[0], 'x',
                num_cells_per_block[1], 'x',
                num_bins, '=',
                tot_els)

            # Print the shape of the HOG Descriptor to see that it matches the above
            print('\nThe HOG Descriptor has shape:', hog_descriptor.shape)