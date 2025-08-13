import glob
import os
import datetime
from multiprocessing import Pool, cpu_count
import argparse


# Import our modular components
from face_processors import (
    left_eye_processor,
    full_face_processor,
    get_date_from_exif,
    get_basename,
)
from happiness_detector import get_happiness_score

# Configuration
OUTPUT_SIZE = (700, 700)  # width x height
DESIRED_EYE_HEIGHT = 130
FRAME_DURATION = 0.2  # seconds per frame
OUTPUT_GIF = "facelapse.gif"
CACHE_DIR = "cache"
NUM_PROCESSES = max(1, int(cpu_count() * 0.75))  # Use 75% of CPU cores

# Update configuration in imported modules
import face_processors

face_processors.OUTPUT_SIZE = OUTPUT_SIZE
face_processors.DESIRED_EYE_HEIGHT = DESIRED_EYE_HEIGHT
face_processors.CACHE_DIR = CACHE_DIR

import happiness_detector

happiness_detector.CACHE_DIR = CACHE_DIR

os.makedirs(CACHE_DIR, exist_ok=True)


def process_image(image_path, frame_index, anchor, draw_date):
    """Main processing function - currently uses full face method."""
    if anchor == "left-eye":
        return left_eye_processor.process(image_path, frame_index, draw_date)
    else:
        return full_face_processor.process(image_path, frame_index, draw_date)


def process_image_wrapper(args):
    """Wrapper function for multiprocessing."""
    path, index, anchor, draw_date = args
    return process_image(path, index, anchor, draw_date)


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
    image_extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp",
        "*.tiff",
        "*.tif",
        "*.webp",
        "*.heic",
        "*.HEIC",
    ]
    all_images = []
    for ext in image_extensions:
        all_images.extend(glob.glob(f"raws/{ext}"))
        all_images.extend(
            glob.glob(f"raws/{ext.upper()}")
        )  # Also check uppercase extensions

    image_paths = [
        path for path in all_images if not os.path.basename(path).startswith("_")
    ]

    if not image_paths:
        print("❌ No images found in raws/ directory")
        return

    parser = argparse.ArgumentParser(description="create GIF of face")
    parser.add_argument(
        "--sort",
        choices=["happiness", "date", "filename"],
        default="date",
        help="Sort order for results (default: date)",
    )
    parser.add_argument(
        "--anchor",
        choices=["left-eye", "face"],
        default="face",
        help="Visual anchor for GIF (default: face)",
    )
    parser.add_argument("--draw-date", action="store_true", help="Draw date on the GIF")
    args = parser.parse_args()
    sort_order = args.sort
    anchor = args.anchor
    draw_date = args.draw_date

    if sort_order == "happiness":
        image_paths.sort(key=get_happiness_score, reverse=False)
    elif sort_order == "filename":
        image_paths.sort()  # Sort by filename (?)
    else:
        image_paths.sort(key=get_image_date)

    print(f"🔍 Found {len(image_paths)} images")

    process_args = [(path, i, anchor, draw_date) for i, path in enumerate(image_paths)]

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
