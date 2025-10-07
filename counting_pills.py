import cv2
import numpy as np

def count_pills_opencv(image_path, display_size=(600, 400)):
    """
    Counts the number of pills in an image using classical computer vision techniques.
    No AI/ML is used, only OpenCV algorithms (thresholding, morphology, watershed).
    
    Parameters:
        image_path (str): Path to the input image.
        display_size (tuple): Resize output window for better display (width, height).
        
    Returns:
        pill_count (int): The number of pills detected in the image.
    """

    # Step 1: Read the input image
    img = cv2.imread(image_path)

    # Step 2: Convert the image to grayscale (removes color information)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 3: Apply Gaussian Blur to reduce noise and smooth edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 4: Apply Otsu's Thresholding to separate pills (foreground) from background
    # Using THRESH_BINARY_INV because pills are usually lighter than the background
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Step 5: Morphological Opening to clean up small noise and keep pill shapes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Step 6: Distance Transform - helps to separate touching pills
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    # Step 7: Threshold the distance transform to find sure foreground regions (pill centers)
    _, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Step 8: Subtract sure foreground from opening to find "unknown" regions (borders between pills)
    unknown = cv2.subtract(opening, sure_fg)

    # Step 9: Connected Components - label different objects in the image
    num_labels, markers = cv2.connectedComponents(sure_fg)

    # Step 10: Watershed segmentation
    # Add 1 to all labels so that background is not 0, but 1
    markers = markers + 1

    # Mark "unknown" regions with 0
    markers[unknown == 255] = 0

    # Apply watershed algorithm: boundaries will be marked with -1
    cv2.watershed(img, markers)

    # Step 11: Count objects
    # Unique markers correspond to different pills
    # Subtract 2 → one for background, one for boundary (-1 regions)
    pill_count = len(np.unique(markers)) - 2

    # Step 12: Draw bounding boxes around detected pills
    result = img.copy()
    contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours, 1):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(result, str(i), (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Step 13: Resize the output window for display (optional)
    if display_size is not None:
        result = cv2.resize(result, display_size)

    # Step 14: Show the final result with detected pills
    cv2.imshow("Detected Pills", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return pill_count


# ----------------- Run the function -----------------
# Change the path to your own image
num_pills = count_pills_opencv("path")

# Print the final pill count
print(f"Number of pills detected: {num_pills}")
