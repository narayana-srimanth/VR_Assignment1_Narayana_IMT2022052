import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

input_folder = "../input_images/"
output_folder = "../output_images/"

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Load images
image_1 = cv2.imread(input_folder + "right_side.jpeg")
image_2 = cv2.imread(input_folder + "left_side.jpeg")

# Convert images to RGB format
image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)

# Convert to grayscale for feature detection
gray_img1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2GRAY)
gray_img2 = cv2.cvtColor(image_2, cv2.COLOR_RGB2GRAY)

# Initialize SIFT feature detector
sift_detector = cv2.SIFT_create()

# Detect keypoints and compute descriptors
kp1, desc1 = sift_detector.detectAndCompute(gray_img1, None)
kp2, desc2 = sift_detector.detectAndCompute(gray_img2, None)

# Draw detected keypoints for visualization
keypoint_img1 = cv2.drawKeypoints(image_1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
keypoint_img2 = cv2.drawKeypoints(image_2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints detected
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Keypoints in Image 1")
plt.imshow(keypoint_img1)

plt.subplot(1, 2, 2)
plt.title("Keypoints in Image 2")
plt.imshow(keypoint_img2)

# Feature matching using BFMatcher
matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches_found = matcher.match(desc1, desc2)

# Sort matches by distance (lower distance = better match)
matches_sorted = sorted(matches_found, key=lambda m: m.distance)

# Visualize the best 10 matches
match_viz = cv2.drawMatches(image_1, kp1, image_2, kp2, matches_sorted[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(12, 6))
plt.imshow(match_viz)
plt.title("Top 10 Matches")

# Select the best 50 matches for homography calculation
best_matches = matches_sorted[:50]

# Extract keypoint coordinates from the best matches
source_points = np.float32([kp1[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
destination_points = np.float32([kp2[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)

# Compute homography matrix with RANSAC
homography_matrix, mask = cv2.findHomography(source_points, destination_points, cv2.RANSAC, 5.0)

# Get image dimensions
img_height, img_width, _ = image_2.shape

# Apply perspective transformation
warped_result = cv2.warpPerspective(image_1, homography_matrix, (img_width * 2, img_height))

# Blend images: overlay image_2 onto the transformed image
warped_result[0:img_height, 0:img_width] = image_2

# Function to remove black regions after stitching
def remove_black_borders(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, threshold = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)

    # Find the bounding rectangle for non-black pixels
    non_black_coords = cv2.findNonZero(threshold)
    x, y, w, h = cv2.boundingRect(non_black_coords)

    return image[y:y+h, x:x+w]

# Apply cropping to remove black borders
final_panorama = remove_black_borders(warped_result)

# Save final stitched image
output_path = os.path.join(output_folder, "stitched_output.jpeg")
cv2.imwrite(output_path, cv2.cvtColor(final_panorama, cv2.COLOR_RGB2BGR))  # Convert back to BGR for saving

# Display the final result
plt.figure(figsize=(12, 6))
plt.imshow(final_panorama)
plt.title("Final Stitched Panorama")
plt.show()

#IMT2022052
#Narayana



