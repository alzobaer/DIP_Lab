from random import random
import cv2
import numpy as np
from matplotlib import pyplot as plt


# apply salt and pepper noise in the image
def add_noise(img):
    row, col = img.shape
    # salt noise
    number_of_pixels = random.randint(300, 10000)  # Pick a random number between 300 and 10000 to be white pixel
    for i in range(number_of_pixels):
        y_coord = random.randint(0, row - 1)  # Pick a random y coordinate
        x_coord = random.randint(0, col - 1)  # Pick a random x coordinate
        img[y_coord][x_coord] = 255  # Color that pixel to white
    # pepper noise
    number_of_pixels = random.randint(300, 10000)  # Pick a random number between 300 and 10000
    for i in range(number_of_pixels):
        y_coord = random.randint(0, row - 1)  # Pick a random y coordinate
        x_coord = random.randint(0, col - 1)  # Pick a random x coordinate
        img[y_coord][x_coord] = 0  # Color that pixel to black
    return img


# calculate Peak signal-to-noise ratio (PSNR)
def cal_psnr(original_img, noisy_image):
    # calculate the mean square error between the original and noisy image
    mse = np.mean((original_img - noisy_image) ** 2)
    max_pixel_value = 255  # maximum possible pixel value
    # Calculate PSNR = 10log((max^2)/mse)
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr


# apply average filtering to the noisy image with kernel size 5
def average_filter(image, kernel_size):
    height, width = image.shape
    filtered_image = image.copy()

    # Define the padding size based on the kernel size
    padding = kernel_size // 2

    for i in range(padding, height - padding):
        for j in range(padding, width - padding):
            # Extract the neighborhood around the pixel
            neighborhood = image[i - padding:i + padding + 1, j - padding:j + padding + 1]

            # Calculate the average value of the neighborhood
            average_value = int(neighborhood.mean())

            # Set the filtered pixel value to the average
            filtered_image[i, j] = average_value
    return filtered_image


# read a grayscale image
original_img = cv2.imread('./images/img4.jpg', cv2.IMREAD_GRAYSCALE)

# make an image with salt and pepper noise
noisy_image = original_img.copy()  # storing the noisy image
noisy_image = add_noise(noisy_image)  # add noise to the original image


# display noisy image
plt.subplot(2, 2, 1)
plt.title('Noisy Image ')
plt.imshow(noisy_image, cmap='gray')

# Define the kernel sizes for average filtering
kernel_sizes = [3, 5, 7]
i = 1

# Apply average filtering with different kernel sizes
for kernel_size in kernel_sizes:
    i = i + 1
    # Apply average filtering
    filtered_image = average_filter(noisy_image, kernel_size)

    # Calculate PSNR
    psnr_values = cal_psnr(original_img, filtered_image)

    # display noisy image
    plt.subplot(2, 2, i)
    plt.title('Noisy Image {kernel_size}x{kernel_size} (PSNR = {:.2f} DB)'.format(psnr_values))
    plt.imshow(noisy_image, cmap='gray')

plt.tight_layout()
plt.show()