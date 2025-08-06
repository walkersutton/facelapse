import glob
import os
import sys
import numpy as np
import cv2
import face_recognition
from PIL import Image, ImageOps, ImageDraw, ImageFont
import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

# Configuration
OUTPUT_SIZE = (1400, 700)  # width x height
DESIRED_EYE_HEIGHT = 130
FRAME_DURATION = 0.2  # seconds per frame
OUTPUT_GIF = "facelapse.gif"
CACHE_DIR = "cache"
FORCE_REPROCESS = "--force" in sys.argv
START_DATE = datetime.date(2025, 6, 21)
SKIP_DATE = datetime.date(2025, 7, 25)
# Font Configuration - Edit these directly
FONT_PATH = "/Library/Fonts/Arial Unicode.ttf"
FONT_SIZE = 40
FONT_COLOR = (255, 255, 255)  # White
TEXT_POSITION = (40, -40)  # (x, y) relative to bottom-right

# Performance settings
NUM_PROCESSES = max(1, int(cpu_count() * 0.75))  # Use 75% of CPU cores

os.makedirs(CACHE_DIR, exist_ok=True)

def get_basename(image_path):
    return os.path.splitext(os.path.basename(image_path))[0]

def get_date_from_index(index):
    current_date = START_DATE
    days_added = 0
    while days_added <= index:
        if current_date == SKIP_DATE:
            current_date += datetime.timedelta(days=1)
            continue
        if days_added == index:
            return current_date
        current_date += datetime.timedelta(days=1)
        days_added += 1

def draw_date_text(image, date_text):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except:
        font = ImageFont.load_default()
    
    # Calculate text position
    x, y = TEXT_POSITION
    if y < 0:  # Negative means from bottom
        y = image.height + y - FONT_SIZE
    
    draw.text((x, y), date_text, font=font, fill=FONT_COLOR)

def process_image(image_path, frame_index):
    basename = get_basename(image_path)
    cached_img_path = os.path.join(CACHE_DIR, f"{basename}.png")

    if os.path.exists(cached_img_path) and not FORCE_REPROCESS:
        cached_img = Image.open(cached_img_path)
        return cached_img.convert('RGB')

    pil_image = Image.open(image_path)
    pil_image = ImageOps.exif_transpose(pil_image)
    img = np.array(pil_image)

    face_landmarks_list = face_recognition.face_landmarks(img)
    if not face_landmarks_list:
        print(f"❌ No face detected in {image_path}")
        return None

    landmarks = face_landmarks_list[0]
    eye_1 = landmarks.get("left_eye")
    eye_2 = landmarks.get("right_eye")

    if not eye_1 or not eye_2:
        print(f"❌ Eyes not found in {image_path}")
        return None

    center_1 = np.mean(eye_1, axis=0)
    center_2 = np.mean(eye_2, axis=0)
    if center_1[0] < center_2[0]:
        left_eye_pts, right_eye_pts = eye_1, eye_2
        left_eye_center, right_eye_center = center_1, center_2
    else:
        left_eye_pts, right_eye_pts = eye_2, eye_1
        left_eye_center, right_eye_center = center_2, center_1

    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dy, dx))
    eye_center = tuple(np.mean([left_eye_center, right_eye_center], axis=0))
    rot_matrix = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    rotated = cv2.warpAffine(
        img,
        rot_matrix,
        (img.shape[1], img.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    # Re-detect landmarks
    face_landmarks_list = face_recognition.face_landmarks(rotated)
    if not face_landmarks_list:
        print(f"❌ No face detected after rotation in {image_path}")
        return None

    landmarks = face_landmarks_list[0]
    left_eye_pts = landmarks.get("left_eye")
    right_eye_pts = landmarks.get("right_eye")
    if not left_eye_pts or not right_eye_pts:
        print(f"❌ Eyes not found after rotation in {image_path}")
        return None

    left_eye_center = np.mean(left_eye_pts, axis=0)
    right_eye_center = np.mean(right_eye_pts, axis=0)

    eye_top = min(p[1] for p in left_eye_pts)
    eye_bottom = max(p[1] for p in left_eye_pts)
    current_eye_height = eye_bottom - eye_top
    if current_eye_height == 0:
        print(f"❌ Invalid eye height in {image_path}")
        return None

    scale = DESIRED_EYE_HEIGHT / current_eye_height
    new_img = cv2.resize(rotated, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    new_left_eye_center = np.array(left_eye_center) * scale
    new_right_eye_center = np.array(right_eye_center) * scale

    target_left_eye_x = int(OUTPUT_SIZE[0] * 0.35)
    target_eye_y = OUTPUT_SIZE[1] // 2

    offset_x = int(target_left_eye_x - new_left_eye_center[0])
    offset_y = int(target_eye_y - new_left_eye_center[1])

    M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    translated = cv2.warpAffine(
        cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR),
        M,
        OUTPUT_SIZE,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    result_img = Image.fromarray(cv2.cvtColor(translated, cv2.COLOR_BGR2RGB))
    date = get_date_from_index(frame_index)

    day_str = f"{date.day:2}"  # space-padded, not zero-padded
    date_text = f"{date.strftime('%B')} {day_str}, {date.year}"

    draw_date_text(result_img, date_text)

    result_img.save(cached_img_path)
    
    # Convert to RGB mode to ensure compatibility with multiprocessing
    result_img = result_img.convert('RGB')
    return result_img

def process_image_wrapper(args):
    """Wrapper function for multiprocessing."""
    path, index = args
    return process_image(path, index)

def main():
    

    
    image_paths = sorted(glob.glob("raws/day*.jpg"))
    
    if not image_paths:
        print("❌ No images found in raws/ directory")
        return
    
    print(f"🔍 Found {len(image_paths)} images")
    
    # Prepare arguments for multiprocessing
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
    
    # Add hold frames at the end
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
