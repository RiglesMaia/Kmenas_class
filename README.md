# K-means Image Classification

This Python script facilitates K-means clustering for image classification using OpenCV and scikit-learn libraries. It allows users to specify parameters such as the image path, number of clusters, desired class, and blur level through a graphical user interface (GUI) created with tkinter.

## Requirements
- Python 3.x
- Libraries:
    - tkinter
    - cv2 (OpenCV)
    - matplotlib
    - numpy
    - scikit-learn

## Usage
1. Clone or download this repository to your local machine.
2. Ensure you have the necessary Python libraries installed (see Requirements section).
3. Run the `main.py` file.
4. The GUI window will prompt you to provide input parameters for image classification.
5. After providing the required inputs and clicking the "OK" button, the script will perform K-means clustering on the input image.
6. It will display the segmented image with the specified number of clusters and extract the desired class region.
7. The script will also show the original image and the cropped region of the desired class.
8. Results such as the number of pixels in the extracted region will be printed in the console.

## Functionality
- Provides a GUI for easy parameter input.
- Applies Gaussian blur to the image to reduce noise.
- Implements the K-means algorithm to segment the image into clusters.
- Extracts the desired class region based on user input.
- Displays both the segmented and original images for visual comparison.
- Outputs the number of pixels in the extracted region.

## Contributions
Contributions to this project are welcome. If you find any bugs or have suggestions for improvements, please feel free to open an issue or submit a pull request.

