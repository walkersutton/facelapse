import os
import numpy as np
import cv2
import face_recognition
from PIL import Image, ImageOps, ImageDraw, ImageFont
import datetime
from multiprocessing import Pool, cpu_count
from PIL.ExifTags import TAGS

# Register HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    print("⚠️  pillow-heif not available. HEIC files will not be supported.")

# Configuration (imported from main.py)
OUTPUT_SIZE = (1000, 1000)  # width x height
DESIRED_EYE_HEIGHT = 130
CACHE_DIR = "ncache"
DRAW_DATE = False
FONT_PATH = "/Library/Fonts/Arial Unicode.ttf"
FONT_SIZE = 40
FONT_COLOR = (255, 255, 255)  # White
TEXT_POSITION = (40, -40)  # (x, y) relative to bottom-right

def get_basename(image_path):
    return os.path.splitext(os.path.basename(image_path))[0]

def get_date_from_exif(image_path):
    """Extract date from image EXIF data."""
    try:
        with Image.open(image_path) as img:
            exif = img._getexif()
            if exif is None:
                return None
            
            for tag_id in exif:
                tag = TAGS.get(tag_id, tag_id)
                data = exif.get(tag_id)
                
                if tag == "DateTimeOriginal":
                    # Parse EXIF date format: "2025:06:21 10:30:00"
                    date_str = data.split()[0]  # Get just the date part
                    year, month, day = map(int, date_str.split(':'))
                    return datetime.date(year, month, day)
                elif tag == "DateTime":
                    # Fallback to regular DateTime if Original not available
                    date_str = data.split()[0]
                    year, month, day = map(int, date_str.split(':'))
                    return datetime.date(year, month, day)
    except Exception as e:
        print(f"⚠️  Could not extract date from {image_path}: {e}")
        return None
    
    return None

def draw_date_text(image, date_text):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except:
        font = ImageFont.load_default()
    
    x, y = TEXT_POSITION
    if y < 0:  # Negative means from bottom
        y = image.height + y - FONT_SIZE
    
    draw.text((x, y), date_text, font=font, fill=FONT_COLOR)

class FaceProcessor:
    """Base class for face processing with shared functionality."""
    
    def __init__(self, method_name):
        self.method_name = method_name
    
    def get_cache_path(self, image_path):
        """Get cache path for this processing method."""
        basename = get_basename(image_path)
        return os.path.join(CACHE_DIR, f"{basename}_{self.method_name}.png")
    
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess image, check cache first."""
        cached_img_path = self.get_cache_path(image_path)
        
        if os.path.exists(cached_img_path):
            cached_img = Image.open(cached_img_path)
            return cached_img.convert('RGB')
        
        pil_image = Image.open(image_path)
        pil_image = ImageOps.exif_transpose(pil_image)
        return np.array(pil_image)
    
    def detect_face_landmarks(self, img):
        """Detect face landmarks and validate eyes."""
        face_landmarks_list = face_recognition.face_landmarks(img)
        if not face_landmarks_list:
            return None, None, None
        
        landmarks = face_landmarks_list[0]
        eye_1 = landmarks.get("left_eye")
        eye_2 = landmarks.get("right_eye")
        
        if not eye_1 or not eye_2:
            return None, None, None
        
        center_1 = np.mean(eye_1, axis=0)
        center_2 = np.mean(eye_2, axis=0)
        
        if center_1[0] < center_2[0]:
            left_eye_pts, right_eye_pts = eye_1, eye_2
            left_eye_center, right_eye_center = center_1, center_2
        else:
            left_eye_pts, right_eye_pts = eye_2, eye_1
            left_eye_center, right_eye_center = center_2, center_1
        
        return left_eye_pts, right_eye_pts, left_eye_center, right_eye_center
    
    def rotate_image(self, img, left_eye_center, right_eye_center):
        """Rotate image to align eyes horizontally."""
        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dy, dx))
        eye_center = tuple(np.mean([left_eye_center, right_eye_center], axis=0))
        rot_matrix = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        
        return cv2.warpAffine(
            img,
            rot_matrix,
            (img.shape[1], img.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
    
    def finalize_image(self, result_img, image_path):
        """Add date text and save to cache."""
        date = get_date_from_exif(image_path)
        if DRAW_DATE and date:
            date_text = f"{date.strftime('%b %d, %Y')}"
            draw_date_text(result_img, date_text)
        
        cached_img_path = self.get_cache_path(image_path)
        result_img.save(cached_img_path)
        
        # Convert to RGB mode to ensure compatibility with multiprocessing
        return result_img.convert('RGB')
    
    def process(self, image_path, frame_index):
        """Main processing method to be implemented by subclasses."""
        raise NotImplementedError


class LeftEyeProcessor(FaceProcessor):
    """Process image focusing on left eye alignment."""
    
    def __init__(self):
        super().__init__("left_eye")
    
    def process(self, image_path, frame_index):
        """Process image focusing on left eye alignment."""
        img = self.load_and_preprocess_image(image_path)
        if isinstance(img, Image.Image):  # Already cached
            return img
        
        # Detect face landmarks
        landmarks = self.detect_face_landmarks(img)
        if landmarks[0] is None:
            print(f"❌ No face detected in {image_path}")
            return None
        
        left_eye_pts, right_eye_pts, left_eye_center, right_eye_center = landmarks
        
        # Rotate image
        rotated = self.rotate_image(img, left_eye_center, right_eye_center)
        
        # Get landmarks again after rotation
        landmarks = self.detect_face_landmarks(rotated)
        if landmarks[0] is None:
            print(f"❌ No face detected after rotation in {image_path}")
            return None
        
        left_eye_pts, right_eye_pts, left_eye_center, right_eye_center = landmarks
        
        # Calculate scale based on left eye height
        eye_top = min(p[1] for p in left_eye_pts)
        eye_bottom = max(p[1] for p in left_eye_pts)
        current_eye_height = eye_bottom - eye_top
        if current_eye_height == 0:
            print(f"❌ Invalid eye height in {image_path}")
            return None
        
        scale = DESIRED_EYE_HEIGHT / current_eye_height
        new_img = cv2.resize(rotated, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        new_left_eye_center = np.array(left_eye_center) * scale
        
        # Position left eye at target location
        target_left_eye_x = int(OUTPUT_SIZE[0] * 0.35)
        target_eye_y = OUTPUT_SIZE[1] // 2
        
        offset_x = int(target_left_eye_x - new_left_eye_center[0])
        offset_y = int(target_eye_y - new_left_eye_center[1])
        
        # Translate image
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
        return self.finalize_image(result_img, image_path)


class FullFaceProcessor(FaceProcessor):
    """Process image centering the full face in the output."""
    
    def __init__(self):
        super().__init__("full_face")
    
    def process(self, image_path, frame_index):
        """Process image centering the full face in the output."""
        img = self.load_and_preprocess_image(image_path)
        if isinstance(img, Image.Image):  # Already cached
            return img
        
        # Detect face landmarks
        landmarks = self.detect_face_landmarks(img)
        if landmarks[0] is None:
            print(f"❌ No face detected in {image_path}")
            return None
        
        left_eye_pts, right_eye_pts, left_eye_center, right_eye_center = landmarks
        
        # Calculate face center and dimensions
        face_center = np.mean([left_eye_center, right_eye_center], axis=0)
        eye_distance = np.linalg.norm(right_eye_center - left_eye_center)
        face_width = eye_distance * 2.5
        face_height = face_width * 1.2
        
        # Calculate scale to fit face in output
        scale_x = OUTPUT_SIZE[0] / face_width
        scale_y = OUTPUT_SIZE[1] / face_height
        scale = min(scale_x, scale_y) * 0.8  # 0.8 to leave some margin
        
        # Rotate image
        rotated = self.rotate_image(img, left_eye_center, right_eye_center)
        
        # Get landmarks again after rotation
        landmarks = self.detect_face_landmarks(rotated)
        if landmarks[0] is None:
            print(f"❌ No face detected after rotation in {image_path}")
            return None
        
        left_eye_pts, right_eye_pts, left_eye_center, right_eye_center = landmarks
        face_center = np.mean([left_eye_center, right_eye_center], axis=0)
        
        # Scale the image
        new_img = cv2.resize(rotated, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        new_face_center = np.array(face_center) * scale
        
        # Center the face in the output
        target_center_x = OUTPUT_SIZE[0] // 2
        target_center_y = OUTPUT_SIZE[1] // 2
        
        offset_x = int(target_center_x - new_face_center[0])
        offset_y = int(target_center_y - new_face_center[1])
        
        # Translate image
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
        return self.finalize_image(result_img, image_path)


# Create processor instances
left_eye_processor = LeftEyeProcessor()
full_face_processor = FullFaceProcessor() 