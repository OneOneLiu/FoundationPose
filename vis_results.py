import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import os
from tqdm import tqdm

def list_files_sorted(directory):
    """ Returns a sorted list of all files in the given directory """
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    files.sort()
    return files

def update_frame(num, rgb_files, track_vis_files, axes, rgb_dir, track_vis_dir, pbar):
    """ Update the frame for animation """
    rgb_img = Image.open(os.path.join(rgb_dir, rgb_files[num]))
    track_vis_img = Image.open(os.path.join(track_vis_dir, track_vis_files[num]))
    axes[0].imshow(rgb_img)
    axes[1].imshow(track_vis_img)
    axes[0].set_title('RGB Image')
    axes[1].set_title('Track Visualization Image')
    axes[0].axis('off')
    axes[1].axis('off')
    
    # Update the tqdm progress bar
    pbar.update(1)

def create_animation(rgb_directory, track_vis_directory):
    rgb_files = list_files_sorted(rgb_directory)
    track_vis_files = list_files_sorted(track_vis_directory)
    total_frames = len(rgb_files)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].set_aspect('equal')
    axes[1].set_aspect('equal')

    
    # Initialize tqdm progress bar
    pbar = tqdm(total=total_frames, desc="Generating Video")
    
    anim = FuncAnimation(fig, update_frame, frames=total_frames, fargs=(rgb_files, track_vis_files, axes, rgb_directory, track_vis_directory, pbar), interval=100, repeat=False)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.05, hspace=0.05)
    
    # Keep a reference to the animation object to prevent it from being garbage collected.
    global _anim
    _anim = anim

    anim.save('output_video.mp4', writer='ffmpeg', dpi=120)
    pbar.close()  # Close the tqdm progress bar after the video has been saved
    print("Video saved successfully.")

# Example usage:
rgb_directory = './demo_data/tube_stack/rgb'
track_vis_directory = './debug/track_vis'
create_animation(rgb_directory, track_vis_directory)
