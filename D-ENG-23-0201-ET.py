import cv2
import numpy as np
import os

def gaussian_filter(image, kernel_size=5, sigma=1.4):
    k = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = k @ k.T
    return cv2.filter2D(image, -1, kernel)

def sobel_filters(image):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Ix = cv2.filter2D(image, -1, Kx)
    Iy = cv2.filter2D(image, -1, Ky)
    gradient_magnitude = np.hypot(Ix, Iy)
    gradient_direction = np.arctan2(Iy, Ix)
    return gradient_magnitude, gradient_direction

def non_maximum_suppression(gradient_magnitude, gradient_direction):
    rows, cols = gradient_magnitude.shape
    suppressed = np.zeros_like(gradient_magnitude)
    angle = np.degrees(gradient_direction) % 180
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            q, r = 255, 255
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = gradient_magnitude[i, j+1]
                r = gradient_magnitude[i, j-1]
            elif 22.5 <= angle[i, j] < 67.5:
                q = gradient_magnitude[i+1, j-1]
                r = gradient_magnitude[i-1, j+1]
            elif 67.5 <= angle[i, j] < 112.5:
                q = gradient_magnitude[i+1, j]
                r = gradient_magnitude[i-1, j]
            elif 112.5 <= angle[i, j] < 157.5:
                q = gradient_magnitude[i-1, j-1]
                r = gradient_magnitude[i+1, j+1]

            if (gradient_magnitude[i, j] >= q) and (gradient_magnitude[i, j] >= r):
                suppressed[i, j] = gradient_magnitude[i, j]
    return suppressed

def thresholding(image, low_threshold, high_threshold):
    strong = 255
    weak = 75
    result = np.zeros_like(image)
    strong_i, strong_j = np.where(image >= high_threshold)
    weak_i, weak_j = np.where((image >= low_threshold) & (image < high_threshold))
    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak
    return result, weak, strong

def edge_tracking(image, weak, strong):
    rows, cols = image.shape
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if image[i, j] == weak:
                if strong in [image[i+1, j-1], image[i+1, j], image[i+1, j+1], image[i, j-1], image[i, j+1], image[i-1, j-1], image[i-1, j], image[i-1, j+1]]:
                    image[i, j] = strong
                else:
                    image[i, j] = 0
    return image

def canny_edge_detection_color(image_path, low_threshold, high_threshold):
    # Load the color image
    color_image = cv2.imread(image_path)
    if color_image is None:
        raise ValueError(f"Image at {image_path} could not be loaded. Check the path.")
    
    # Convert the color image to grayscale
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    
    # Process the grayscale image for edge detection
    smoothed = gaussian_filter(gray_image)
    gradient_magnitude, gradient_direction = sobel_filters(smoothed)
    non_max_suppressed = non_maximum_suppression(gradient_magnitude, gradient_direction)
    thresholded, weak, strong = thresholding(non_max_suppressed, low_threshold, high_threshold)
    edges = edge_tracking(thresholded, weak, strong)
    
    # Convert edges to uint8 (fix for cv2.COLOR_GRAY2BGR compatibility)
    edges = edges.astype(np.uint8)

    # Convert edges to 3-channel format for overlay
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Overlay edges onto the original color image
    result = cv2.addWeighted(color_image, 0.8, edges_color, 0.2, 0)
    return result, edges

# Paths and thresholds
image_path = r"C:\Users\shine\Desktop\prac 3\lotus-flower-top-view-on-260nw-2505626641.jpg"
output_path_edges = r"C:\Users\shine\Desktop\prac 3\output_edges.jpg"
output_path_overlay = r"C:\Users\shine\Desktop\prac 3\output_overlay.jpg"
low_threshold = 50
high_threshold = 150

# Perform edge detection
try:
    result_image, edges = canny_edge_detection_color(image_path, low_threshold, high_threshold)

    # Save results
    cv2.imwrite(output_path_edges, edges)
    cv2.imwrite(output_path_overlay, result_image)
    print("Outputs saved successfully.")
except Exception as e:
    print(f"Error: {e}")
