import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Define input and output directories
input_folder = "../input_images/"
output_folder = "../output_images/"
os.makedirs(output_folder, exist_ok=True)

def load_and_convert(image_path):
    """Loads an image and converts it to RGB & grayscale."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return image_rgb, image_gray

def detect_and_describe(image_rgb, image_gray, image_name):
    """Detects keypoints and computes descriptors using SIFT."""
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image_gray, None)

    # Draw and save keypoints visualization
    keypoint_image = cv2.drawKeypoints(image_rgb, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    keypoint_path = os.path.join(output_folder, f"keypoints_{image_name}.jpeg")
    cv2.imwrite(keypoint_path, cv2.cvtColor(keypoint_image, cv2.COLOR_RGB2BGR))

    return keypoints, descriptors, keypoint_path

def match_features(desc1, desc2):
    """Matches features using BFMatcher with cross-check."""
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(desc1, desc2)
    return sorted(matches, key=lambda x: x.distance)

def visualize_matches(image1, kp1, image2, kp2, matches):
    """Draws and saves the best matches."""
    match_image = cv2.drawMatches(image1, kp1, image2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    match_path = os.path.join(output_folder, "image_matches.jpeg")
    cv2.imwrite(match_path, cv2.cvtColor(match_image, cv2.COLOR_RGB2BGR))
    return match_path

def compute_homography(kp1, kp2, matches, max_matches=50):
    """Computes the homography matrix using the best matches."""
    if len(matches) < 4:
        raise ValueError("Not enough matches to compute homography!")

    best_matches = matches[:max_matches]
    src_pts = np.float32([kp1[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H

def warp_and_blend(image1, image2, H):
    """Warps image1 to align with image2 and blends them."""
    h2, w2, _ = image2.shape
    warped_image = cv2.warpPerspective(image1, H, (w2 * 2, h2))
    warped_image[0:h2, 0:w2] = image2  # Overlay second image
    return warped_image

def remove_black_borders(image):
    """Crops out black regions from the stitched image."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y+h, x:x+w]

# Load images
image1, gray1 = load_and_convert(os.path.join(input_folder, "right_side.jpeg"))
image2, gray2 = load_and_convert(os.path.join(input_folder, "left_side.jpeg"))

# Detect features and save keypoints
kp1, desc1, keypoint_img1 = detect_and_describe(image1, gray1, "image1")
kp2, desc2, keypoint_img2 = detect_and_describe(image2, gray2, "image2")

# Match features
matches = match_features(desc1, desc2)

# Visualize and save matches
match_img = visualize_matches(image1, kp1, image2, kp2, matches)

# Compute homography
H = compute_homography(kp1, kp2, matches)

# Warp and blend images
stitched_image = warp_and_blend(image1, image2, H)

# Crop black regions
final_panorama = remove_black_borders(stitched_image)

# Save final stitched image
stitched_output_path = os.path.join(output_folder, "stitched_output.jpeg")
cv2.imwrite(stitched_output_path, cv2.cvtColor(final_panorama, cv2.COLOR_RGB2BGR))

# Display all results
plt.figure(figsize=(15, 8))

plt.subplot(2, 2, 1)
plt.imshow(cv2.imread(keypoint_img1))
plt.title("Keypoints in Image 1")

plt.subplot(2, 2, 2)
plt.imshow(cv2.imread(keypoint_img2))
plt.title("Keypoints in Image 2")

plt.subplot(2, 2, 3)
plt.imshow(cv2.imread(match_img))
plt.title("Feature Matches")

plt.subplot(2, 2, 4)
plt.imshow(final_panorama)
plt.title("Final Stitched Panorama")

plt.show()

