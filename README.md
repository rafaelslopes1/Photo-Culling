# Image Culling Pipeline - Enhanced Version

A Python-based image culling pipeline that automatically processes and **organizes** images into categorized folders for easy manual review.

## Key Features

- **Intelligent Organization:** Instead of just keeping "good" images, this script now organizes ALL images into categorized folders, making manual review much easier.
- **Duplicate Detection:** Uses perceptual hashing to identify and separate duplicate images.
- **Quality Assessment:** Filters images based on blurriness and brightness levels.
- **NSFW Filtering:** Optional NSFW content detection and separation.
- **Smart Ranking:** Quality-based ranking of approved images with detailed scoring.

## Output Structure

The script creates an organized folder structure for easy review:

```
output/
├── selected/     📸 High-quality images (ranked: 001_85.23_IMG_0001.JPG)
├── duplicates/   🔄 Duplicate images detected
├── blurry/       💫 Images that are too blurry  
├── low_light/    🌑 Images that are too dark
├── nsfw/         🔞 NSFW content (if detection enabled)
└── failed/       ❌ Images that failed processing
```

**Benefits:**
- **Easy Recovery:** Quickly review and recover images from any category
- **Manual Override:** Final decision remains with you
- **Quality Insights:** Understand why images were categorized
- **Batch Processing:** Process thousands of images efficientlylling Pipeline

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
```bash
git clone https://github.com/rawatrob/Photo-Culling.git


```
```bash
cd Photo-Culling

```
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


---
## License

This project is licensed under the MIT License -