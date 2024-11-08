# Name: Jazwaur Ankrah
# Number: 001027898
# Machine Vision Assignment 3: Morphology

import numpy as np
import cv2
import matplotlib.pyplot as plt

def morphology():
    #get image
    image = cv2.imread('morphology.png')
    #color scale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # define kernel 
    kernel = np.ones((5, 5), np.uint8)

    # apply erosion
    erosion = cv2.erode(gray_image, kernel, iterations=1)

    # apply median filtering
    median_filtered = cv2.medianBlur(gray_image, 5)
    
    # display plot
    titles = ['Original Image', 'Erosion', 'Original Image', 'Median Filtered']
    images = [gray_image, erosion, gray_image, median_filtered]
    letters = ['(a) ', '(b) ', '(c) ', '(d) ']

    for i in range(4):
        plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
        plt.title(letters[i] + titles[i])
        plt.xticks([]), plt.yticks([])

    plt.show()

def fingerprint():
    #get image
    image = cv2.imread('fingerprint_BW.png')
    #color scale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # define kernel 
    kernel = np.ones((5, 5), np.uint8)

    # apply erosion
    erosion = cv2.erode(gray_image, kernel, iterations=1)

    # apply median filtering
    median_filtered = cv2.medianBlur(gray_image, 5)
    
    # display plot
    titles = ['Original Image', 'Erosion', 'Original Image', 'Median Filtered']
    images = [gray_image, erosion, gray_image, median_filtered]
    letters = ['(a) ', '(b) ', '(c) ', '(d) ']

    for i in range(4):
        plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
        plt.title(letters[i] + titles[i])
        plt.xticks([]), plt.yticks([])

    plt.show()

def cell():
    #get image
    image = cv2.imread('cell.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #binary thresholding
    binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    #get kernel
    kernel = np.ones((3, 3), np.uint8)

    #remove small noise
    opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)

    #dilation
    dilated = cv2.dilate(opening, kernel, iterations=3)

    #distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    #find background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    #find unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    #label markers
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 0] = 0

    #watershed (better segmentation)
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]

    #contrours
    contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #if the contour area is greater than 0, keep it
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 0]

    #get # of cells and size of each cell
    num_cells = len(filtered_contours)
    cell_sizes = [cv2.contourArea(contour) for contour in filtered_contours]

    #mark contours on the original image
    contour_image = image.copy()
    cv2.drawContours(contour_image, filtered_contours, -1, (0, 255, 0), 2)

    #print results
    print(f'Total number of cells: {num_cells}')
    print(f'Sizes of each cell: {cell_sizes}')

    #plot data
    titles = ['Original Image', 'Binary Image', 'Dilated Image', 'Contours']
    images = [image, binary_image, dilated, contour_image]
    letters = ['(a) ', '(b) ', '(c) ', '(d) ']

    plt.figure(figsize=(12, 8))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(letters[i] + titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()

#Main
morphology()
fingerprint()
cell()