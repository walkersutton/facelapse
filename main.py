import glob
import os
import sys
import datetime
from multiprocessing import Pool, cpu_count

# Import our modular components
from face_processors import (
    left_eye_processor, 
    full_face_processor,
    get_date_from_exif,
    get_basename
)
from happiness_detector import get_happiness_score

# Configuration
OUTPUT_SIZE = (1000, 1000)  # width x height
DESIRED_EYE_HEIGHT = 130
FRAME_DURATION = 0.2  # seconds per frame
OUTPUT_GIF = "nfacelapse.gif"
CACHE_DIR = "ncache"
FORCE_REPROCESS = "--force" in sys.argv
DRAW_DATE = False
NUM_PROCESSES = max(1, int(cpu_count() * 0.75))  # Use 75% of CPU cores

# Update configuration in imported modules
import face_processors
face_processors.OUTPUT_SIZE = OUTPUT_SIZE
face_processors.DESIRED_EYE_HEIGHT = DESIRED_EYE_HEIGHT
face_processors.CACHE_DIR = CACHE_DIR
face_processors.FORCE_REPROCESS = FORCE_REPROCESS
face_processors.DRAW_DATE = DRAW_DATE

import happiness_detector
happiness_detector.CACHE_DIR = CACHE_DIR
happiness_detector.FORCE_REPROCESS = FORCE_REPROCESS

os.makedirs(CACHE_DIR, exist_ok=True)

def process_image(image_path, frame_index):
    """Main processing function - currently uses full face method."""
    return full_face_processor.process(image_path, frame_index)
    # return left_eye_processor.process(image_path, frame_index)

def process_image_wrapper(args):
    """Wrapper function for multiprocessing."""
    path, index = args
    return process_image(path, index)

def get_image_date(image_path):
    """Get the date when the image was taken from EXIF data."""
    date = get_date_from_exif(image_path)
    if date is None:
        # If no EXIF date, use file modification time as fallback
        mtime = os.path.getmtime(image_path)
        return datetime.date.fromtimestamp(mtime)
    return date

def main():
    # Get all image files, excluding those with "_" prefix
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif", "*.webp", "*.heic", "*.HEIC"]
    all_images = []
    for ext in image_extensions:
        all_images.extend(glob.glob(f"rawnathan/{ext}"))
        all_images.extend(glob.glob(f"rawnathan/{ext.upper()}"))  # Also check uppercase extensions
    
    image_paths = [path for path in all_images if not os.path.basename(path).startswith("_")]
    
    if not image_paths:
        print("❌ No images found in rawnathan/ directory")
        return
    
    # Sort images based on configuration
    image_paths.sort(key=get_happiness_score, reverse=True)
    # image_paths.sort(key=get_image_date)
    # image_paths.sort()  # Sort by filename

    
    print(f"🔍 Found {len(image_paths)} images")
    
    process_args = [(path, i) for i, path in enumerate(image_paths)]
    
    print(f"🚀 Using {NUM_PROCESSES} processes for parallel processing")
    
    # Process images in parallel
    frames = []
    with Pool(processes=NUM_PROCESSES) as pool:
        # Use imap to get progress updates
        results = pool.imap(process_image_wrapper, process_args)
        
        for i, result in enumerate(results, 1):
            if result is not None:
                frames.append(result)
            print(f"📸 Processed {i}/{len(image_paths)} images")
    
    if not frames:
        print("❌ No frames processed successfully.")
        return
    
    # Hold last frame for a bit
    if frames:
        frames.extend([frames[-1]] * 10)
    
    print(f"📽️ Creating GIF with {len(frames)} frames...")
    try:
        frames[0].save(
            OUTPUT_GIF,
            save_all=True,
            append_images=frames[1:],
            duration=int(FRAME_DURATION * 1000),
            loop=0,
            optimize=False,  # Disable optimization to preserve quality
        )
        print(f"✅ Saved GIF to {OUTPUT_GIF}")
    except Exception as e:
        print(f"❌ Error saving GIF: {e}")
        return

if __name__ == "__main__":
    main()
