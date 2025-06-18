import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def to_grayscale(img):
    """
    Convert an RGB or RGBA image to grayscale using standard luminance formula.
    If image is already single-channel, return as is.
    """
    if img.ndim == 3:
        # Handle RGBA by dropping alpha
        if img.shape[2] == 4:
            img = img[..., :3]
        # Use luminance conversion
        return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    return img

def main():
    # Read images
    img1 = mpimg.imread("demo_data/tube75/depth/depth_image_1750188649_228479601.png")
    img2 = mpimg.imread("demo_data/mustard0/depth/1581120424100262102.png")

    # Convert to grayscale
    gray1 = to_grayscale(img1)
    gray2 = to_grayscale(img2)

    # Create a 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Top row: images
    axes[0, 0].imshow(img1)
    axes[0, 0].axis("off")
    axes[0, 0].set_title("Image 1")

    axes[0, 1].imshow(img2)
    axes[0, 1].axis("off")
    axes[0, 1].set_title("Image 2")

    # Bottom row: histograms
    axes[1, 0].hist(gray1.ravel(), bins=256)
    axes[1, 0].set_title("Histogram of Image 1")
    axes[1, 0].set_xlabel("Intensity")
    axes[1, 0].set_ylabel("Frequency")

    axes[1, 1].hist(gray2.ravel(), bins=256)
    axes[1, 1].set_title("Histogram of Image 2")
    axes[1, 1].set_xlabel("Intensity")
    axes[1, 1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

