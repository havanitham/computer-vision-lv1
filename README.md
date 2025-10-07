# Pill Counting using OpenCV

This project detects and counts the number of pills in an image using **classical computer vision techniques** with OpenCV.
It **does not use AI or machine learning**. Instead, it relies on image processing algorithms like **thresholding, morphology, distance transform, and watershed segmentation**.

## Features

* Works on pill images of different **shapes, sizes, and colors**.
* Uses a **single algorithm** for all images (no hardcoding per image).
* Visualizes each detected pill with a **bounding box and number label**.
* Follows the requirement: **no AI used**.

## Techniques Used

1. **Grayscale conversion** – simplifies the image.
2. **Gaussian blur** – reduces noise.
3. **Otsu’s thresholding** – separates pills from the background.
4. **Morphological operations** – removes small unwanted noise.
5. **Distance transform + Watershed** – separates overlapping or touching pills.
6. **Contour detection** – draws bounding boxes and labels.


## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/pill-counting-opencv.git
cd pill-counting-opencv
```

### 2. Install requirements

Make sure you have Python installed. Then install dependencies:

```bash
pip install opencv-python numpy
```

### 3. Run the script

```bash
python count_pills.py
```
