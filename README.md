# Image Culling Pipeline

A Python-based image culling pipeline that automatically processes a folder of images by:

- **Removing duplicate images** using perceptual hashing.
- **Filtering out low-quality images** based on blurriness (using the variance of the Laplacian) and low brightness.
- **Optionally filtering NSFW images** using a pre-trained NSFW detection model.
- **Ranking and selecting the “best” images** by combining a quality score (sharpness + brightness).

This script is ideal for projects where you need to curate a collection of images by discarding duplicates and images that don’t meet your quality standards.

## Features

- **Duplicate Detection:**  
  Leverages the `imagehash` library to identify and remove duplicate images.

- **Quality Assessment:**  
  - **Blurriness Detection:** Uses OpenCV to compute the variance of the Laplacian.  
  - **Brightness Evaluation:** Analyzes the average brightness from the HSV color space.
  
- **NSFW Filtering (Optional):**  
  Integrates with an NSFW detection model (if provided) to filter out inappropriate content.

- **Ranking:**  
  Computes a combined quality score (sharpness + brightness) to rank the images, and outputs the best images in order.

## Requirements

- Python 3.x
- [OpenCV](https://opencv.org/) (`opencv-python`)
- [Pillow](https://python-pillow.org/)
- [ImageHash](https://github.com/JohannesBuchner/imagehash)
- *(Optional)* [nsfw_detector](https://github.com/infinitered/nsfwjs) for NSFW filtering

### Install Dependencies

Install the necessary packages using pip:

```bash
pip install opencv-python pillow imagehash
```
Basic Command
```bash
python image_culling.py <input_folder> <output_folder>
```

With NSFW Filtering

```bash
python image_culling.py <input_folder> <output_folder> --nsfw_model path/to/nsfw_model.h5

```

Example
```bash
python image_culling.py ./input_images ./output_images --blur_threshold 100 --brightness_threshold 50 --nsfw_threshold 0.7
```