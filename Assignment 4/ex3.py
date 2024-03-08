import sys
import math

import cv2
import numpy as np


def plot_pyramid(pyramid, include_text=False, equalize=False):
  # Sort pyramids from smallest to largest
  pyramid.sort(key=lambda image: image.shape[0] * image.shape[1], reverse=True)

  width_margin = 20
  height_margin = 0

  max_height = max(image.shape[0] for image in pyramid)
  max_height += height_margin
  total_width = sum(image.shape[1] for image in pyramid)
  total_width += width_margin * (len(pyramid) - 1)

  output = np.zeros((max_height, total_width, 3), dtype=np.uint8)
  offset = 0

  for index, image in enumerate(pyramid):
    height, width, _ = image.shape

    if equalize:
      image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255

    output[max_height-height-height_margin:max_height-height_margin, offset:offset + width, :] = image

    if include_text:
      cv2.putText(output, f"Level {index}", (offset + 10, max_height - 10),
                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

    offset += width + width_margin

  return output

def plot_process(image1, image2, mask, result):
  images = [image1, image2, mask, result]
  names = ["Image 1", "Image 2", "Mask", "Result"]

  width_margin = 20
  height_margin = 50

  max_height = max(image.shape[0] for image in images)
  max_height += height_margin
  total_width = sum(image.shape[1] for image in images)
  total_width += width_margin * (3)

  if len(image1.shape) == 3:
    output = np.zeros((max_height, total_width, 3), dtype=np.uint8)
  else:
    output = np.zeros((max_height, total_width), dtype=np.uint8)

  offset = 0

  for index, image in enumerate(images):
    height, width = image.shape[:2]

    if len(image1.shape) == 3:
      output[max_height-height-height_margin:max_height-height_margin, offset:offset + width, :] = image
    else:
      output[max_height-height-height_margin:max_height-height_margin, offset:offset + width] = image

    cv2.putText(output, names[index], (offset + 10, max_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

    offset += width + width_margin

  return output


def compute_gaussian_pyramid(image):
  """
  Compute the Gaussian pyramid of a given image

  Args:
    image (np.array): The input image

  Returns:
    list: A list of images representing the Gaussian pyramid
          The depth of the pyramid (= length of the list) is equal to
          the minimum number of levels that can be computed for the given 
          image, i.e. min(log2(image.shape[0]), log2(image.shape[1]))
  """

  depth = min(4, math.floor(min(math.log2(image.shape[0]), 
                        math.log2(image.shape[1]))))

  pyramid = [image.copy()]

  for i in range(depth):
    level = np.float32(cv2.pyrDown(pyramid[i]))
    
    pyramid.append(level)

  return pyramid

def compute_laplacian_pyramid(gaussian_pyramid):
  """
  Compute the Laplacian pyramid of a given Gaussian pyramid

  Args:
    gaussian_pyramid (list): A list of images representing the Gaussian pyramid

  Returns:
    list: A list of images representing the Laplacian pyramid
  """

  pyramid = [gaussian_pyramid[-1]]

  for i in range(len(gaussian_pyramid) - 1, 0, -1):
    shape = (gaussian_pyramid[i-1].shape[1], gaussian_pyramid[i-1].shape[0])

    # Resizing the expanded image to the shape of the next level
    expanded = cv2.resize(cv2.pyrUp(gaussian_pyramid[i]), shape)
    laplacian = gaussian_pyramid[i-1] - expanded

    pyramid.append(laplacian)

  return pyramid


def blend_laplacian_pyramids(laplacian_pyramid_1, laplacian_pyramid_2, 
                             mask_gaussian_pyramid): 
  """
  Blend two Laplacian pyramids using a mask

  Args:
    laplacian_pyramid_1 (list): A list of images representing the Laplacian pyramid of the first image
    laplacian_pyramid_2 (list): A list of images representing the Laplacian pyramid of the second image
    mask_gaussian_pyramid (list): A list of images representing the Gaussian pyramid of the mask
  
  Returns:
    list: A list of images representing the blended Laplacian pyramid
  """
  
  blended_pyramid = []
  
  for lap1, lap2, mask in zip(laplacian_pyramid_1, laplacian_pyramid_2, 
                              mask_gaussian_pyramid):
    blended_pyramid.append(lap1 * mask + lap2 * (1 - mask))
  
  return blended_pyramid

def create_gaussian_mask(height, width, sigma):
  mask = np.zeros((height, width), dtype=np.float32)
    
  center_row = height // 2
  center_col = width // 2

  for (i, j), _ in np.ndenumerate(mask):
    numerator = np.square(i - center_row) + np.square(j - center_col)
    denominator = 2 * np.square(sigma)
    mask[i, j] = np.exp(-numerator / denominator)

  return mask


def reconstruct_pyramid(pyramid):
  """
  Reconstruct the image from its pyramid

  Args:
    pyramid (list): A list of images representing the pyramid

  Returns:
    list: A list of images representing the reconstructed pyramid
          The image at the last index is the fully, reconstructed image
  """
  
  reconstructed = pyramid[0]
  reconstructed_pyramid = [pyramid[0]]
  
  for i in range(len(pyramid) - 1):
    shape = (pyramid[i+1].shape[1], pyramid[i+1].shape[0])

    expanded = cv2.resize(cv2.pyrUp(reconstructed), shape)
    reconstructed = expanded + pyramid[i+1]

    reconstructed_pyramid.append(reconstructed)

  return reconstructed_pyramid


def pyramid_blending(image1, image2, mask):
  mask = np.round(mask).astype(np.uint8)

  mask_gaussian_pyramid = compute_gaussian_pyramid(mask)
  mask_gaussian_pyramid.reverse()
  
  image1_gaussian_pyramid = compute_gaussian_pyramid(image1)
  image2_gaussian_pyramid = compute_gaussian_pyramid(image2)
  image1_laplacian_pyramid = compute_laplacian_pyramid(image1_gaussian_pyramid)
  image2_laplacian_pyramid = compute_laplacian_pyramid(image2_gaussian_pyramid)

  blended_laplacian_pyramid = blend_laplacian_pyramids(
    image1_laplacian_pyramid,
    image2_laplacian_pyramid,
    mask_gaussian_pyramid
  )
                    
  reconstructed_pyramid = reconstruct_pyramid(
    blended_laplacian_pyramid
  )

  blended_image = reconstructed_pyramid[-1]

  return blended_image

