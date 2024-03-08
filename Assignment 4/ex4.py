import numpy as np

import cv2


KNN_THRESHOLD = 0.75
RANSAC_REPROJECTION_THRESHOLD = 5.0


def import_image(image_path):
  return cv2.imread(image_path)


def save_image(image, image_name):
  cv2.imwrite(image_name, image)


def save_image_with_feature_points(image, feature_points, image_name):
  image = cv2.drawKeypoints(
    image, 
    feature_points, 
    image,
    # flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
  )

  cv2.imwrite(image_name, image)


def save_image_with_matches(image1, feature_points1, image2, feature_points2, matches, image_name):
  image_matches = cv2.drawMatches(
    image1, feature_points1,
    image2, feature_points2,
    matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
  )

  cv2.imwrite(image_name, image_matches)


def compute_feature_points_and_descriptors(image):
  """
  Compute feature points and descriptors for the given image using SIFT

  Args:
    image: The image to compute the feature points and descriptors for

  Returns:
    feature_points: The list of feature points
    descriptors: The list of descriptors
  """

  sift = cv2.SIFT_create()

  return sift.detectAndCompute(image, None)    


def compute_good_matches(descriptors1, descriptors2):
  """
  Compute the good matches between the descriptors of two images
  using the Brute Force Matcher with L2 Norm and Cross Check
  - Cross Check = both vectors agree they are the best for each other
  - KNN = we want to find the best 2 matches for each descriptor
  More info at https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html

  Args:
    descriptors1: The list of descriptors for the first image
    descriptors2: The list of descriptors for the second image

  Returns:
    good_matches: The list of good matches
  """

  matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

  matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
  good_indices = np.where([m.distance / n.distance < KNN_THRESHOLD for m,n in matches])
  good_matches = [m for m,n in matches if m.distance / n.distance < KNN_THRESHOLD]

  print(f'Number of matches: {len(matches)}')
  print(f'Number of good matches: {len(good_matches)}')

  return good_matches, good_indices


def compute_homography(feature_points1, feature_points2, good_matches, inverse=True):
  """
  Compute the homography matrix using the source and destination points,
  and the list of good matches. We compute the homography with RANSAC.
  The homography matrix is used to warp the second image to the first one.

  Args:
    feature_points1: The list of feature points for the first image
    feature_points2: The list of feature points for the second image
    good_matches: The list of good matches
    inverse: Whether to compute the inverse homography matrix

  Returns:
    M: The homography matrix
  """

  source = np.float32([feature_points1[m.queryIdx].pt for m in good_matches])
  source = source.reshape(-1,1,2)
  destination = np.float32([feature_points2[m.trainIdx].pt for m in good_matches])
  destination = destination.reshape(-1,1,2)
  
  M, _ = cv2.findHomography(source, destination, cv2.RANSAC, RANSAC_REPROJECTION_THRESHOLD)

  if inverse:
    M = np.linalg.inv(M)

  return M


def generate_warped_mask(image2):
  """
  Generate the mask for the second image after warping it to the first one.
  It uses the second image to generate an initial mask, then blurs it and 
  turns it into a binary mask.

  Args:
    image2: The warped, second image

  Returns:
    mask: The mask for the second image
  """

  mask = np.zeros_like(image2, dtype=np.float32)
  
  indexes = np.where((image2[:,:,0] != 0) | (image2[:,:,1] != 0) | (image2[:,:,2] != 0))
  mask[indexes] = [1, 1, 1]

  mask = cv2.GaussianBlur(mask, (7, 7), 0)

  indexes = np.where((mask[:,:,0] != 1) & (mask[:,:,1] != 1) & (mask[:,:,2] != 1))
  mask[indexes] = [0, 0, 0]

  return mask

if __name__ == "__main__":
  # img1 = 'input/desert_low_res.jpg'
  # img2 = 'input/desert_high_res.png'

  img1 = 'input/lake_low_res.jpg'
  img2 = 'input/lake_high_res.png'

  image1 = import_image(img1)
  image2 = import_image(img2)

  save_image(image1, "image1.jpg")
  save_image(image2, "image2.jpg")

  # Compute feature points and descriptors for both images using SIFT
  feature_points1, descriptors1 = compute_feature_points_and_descriptors(image1)
  feature_points2, descriptors2 = compute_feature_points_and_descriptors(image2)

  # save_image_with_feature_points(image1, feature_points1, "image1.jpg")
  # save_image_with_feature_points(image2, feature_points2, "image2.jpg")

  good_matches, good_indices = compute_good_matches(descriptors1, descriptors2)

  # save_image_with_feature_points(image1, np.take(feature_points1, good_indices)[0], "image1.jpg")
  # save_image_with_feature_points(image2, np.take(feature_points2, good_indices)[0], "image2.jpg")

  # save_image_with_matches(image1, feature_points1, image2, feature_points2, good_matches, "matches.jpg" )

  M = compute_homography(feature_points1, feature_points2, good_matches, 
                         inverse=True)
  
  # Re-import the images (as normallized floats) in order to:
  # 1. avoid the changes made by the previous function
  # 2. be able to mask the images correctly
  image1 = import_image(img1).astype(np.float32) / 255
  image2 = import_image(img2).astype(np.float32) / 255

  # Applying the inverse homography matrix to warp the second image to the first
  image2 = cv2.warpPerspective(image2, M, dsize=(image1.shape[1], image1.shape[0]))
  
  save_image(image2, "warped.jpg")

  # Generate the final image
  mask = generate_warped_mask(image2)
  # cv2.imwrite('mask.jpg', (mask * 255).astype(np.uint8))
  image = image1 * (1 - mask) + image2 * mask

  # Generate the final image using ex3
  # from ex3 import pyramid_blending
  # image = pyramid_blending(image2, image1, mask)

  cv2.imwrite('final.jpg', (image * 255).astype(np.uint8))