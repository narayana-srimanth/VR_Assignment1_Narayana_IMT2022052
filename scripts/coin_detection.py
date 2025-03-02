import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, color
import os

def count_coins(image_path, output_folder):
    # Load the image
    img_original = cv2.imread(image_path)

    # Create a copy for contour drawing
    img_contours = img_original.copy()

    # Convert images from BGR to RGB (for visualization)
    img_contours = cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB)
    img_display = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

    # Convert to grayscale
    img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    
    # Apply median blur
    img_blurred = cv2.medianBlur(img_gray, 7)

    # Apply thresholding for binary conversion
    _, img_binary = cv2.threshold(img_blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological operation to clean up noise
    morph_kernel = np.ones((3, 3), np.uint8)
    img_cleaned = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, morph_kernel, iterations=2)

    # Labeling connected components for segmentation
    labeled_img = measure.label(img_cleaned, connectivity=2)
    img_segmented = color.label2rgb(labeled_img, bg_label=0)

    # Find contours
    contours, _ = cv2.findContours(img_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw detected contours on the image
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 4)

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Save detected images
    detected_path = os.path.join(output_folder, "detected_coins.jpeg")
    segmented_path = os.path.join(output_folder, "segmented_coins.jpeg")

    cv2.imwrite(detected_path, cv2.cvtColor(img_contours, cv2.COLOR_RGB2BGR))
    cv2.imwrite(segmented_path, cv2.cvtColor((img_segmented * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    # Display results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img_display)

    plt.subplot(1, 2, 2)
    plt.title("Detected Coins")
    plt.imshow(img_contours)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img_display)

    plt.subplot(1, 2, 2)
    plt.title("Segmented Image")
    plt.imshow(img_segmented)
    plt.show()

    return len(contours)

# Define input and output paths
input_image = "../input_images/coins1.jpeg"
output_directory = "../output_images/"

# Run detection
coins_detected = count_coins(input_image, output_directory)
print(f"Detected {coins_detected} coins")

#IMT2022052
#NARAYANA
