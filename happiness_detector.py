import os
import numpy as np
from PIL import Image, ImageOps
import json
from deepface import DeepFace

# Register HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    print("⚠️  pillow-heif not available. HEIC files will not be supported.")

# Configuration
CACHE_DIR = "ncache"
HAPPINESS_CACHE_FILE = os.path.join(CACHE_DIR, "happiness_scores.json")

def analyze_happiness(image_path):
    """Analyze happiness level in an image using DeepFace emotion detection."""
    try:
        # Use DeepFace to analyze emotions
        result = DeepFace.analyze(
            img_path=image_path,
            actions=['emotion'],
            enforce_detection=False,  # Don't fail if face detection fails
            detector_backend='opencv'  # Faster than default
        )
        
        # DeepFace returns a list if multiple faces, or dict if single face
        if isinstance(result, list):
            if not result:
                return 0.0  # No faces detected
            emotions = result[0]['emotion']
        else:
            emotions = result['emotion']
        
        # Get happiness score from DeepFace emotions
        # DeepFace provides: angry, disgust, fear, happy, sad, surprise, neutral
        happy_score = float(emotions.get('happy', 0))  # Convert to Python float
        neutral_score = float(emotions.get('neutral', 0))  # Convert to Python float
        
        # Calculate weighted happiness score
        # Pure happiness gets highest score, neutral gets medium score
        happiness_score = (happy_score * 1.0 + neutral_score * 0.3) / 100.0
        
        # Clamp to 0-1 range
        return max(0.0, min(1.0, happiness_score))
        
    except Exception as e:
        print(f"⚠️  Could not analyze happiness for {image_path}: {e}")
        return 0.0

def get_happiness_score(image_path):
    """Get happiness score for an image, using cache if available."""
    # Load cached scores
    happiness_scores = {}
    if os.path.exists(HAPPINESS_CACHE_FILE):
        try:
            with open(HAPPINESS_CACHE_FILE, 'r') as f:
                happiness_scores = json.load(f)
        except:
            happiness_scores = {}
    
    # Check if we have a cached score
    if image_path in happiness_scores:
        return happiness_scores[image_path]
    
    # Calculate happiness score
    score = analyze_happiness(image_path)
    
    # Cache the score
    happiness_scores[image_path] = score
    try:
        with open(HAPPINESS_CACHE_FILE, 'w') as f:
            json.dump(happiness_scores, f)
    except Exception as e:
        print(f"⚠️  Could not save happiness scores: {e}")
    
    return score 