import cv2
import numpy as np
from PIL import Image
import io
import base64

class ImageProcessor:
    @staticmethod
    def allowed_file(filename, allowed_extensions):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in allowed_extensions
    
    @staticmethod
    def load_and_preprocess_image(file_stream, target_size=(224, 224)):
        """Load and preprocess uploaded image for analysis"""
        try:
            # Convert file stream to numpy array
            file_bytes = np.frombuffer(file_stream.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Could not decode image")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image
            image = cv2.resize(image, target_size)
            
            # Normalize to 0-1
            image = image.astype(np.float32) / 255.0
            
            return image
            
        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")
    
    @staticmethod
    def numpy_to_base64(image_array):
        """Convert numpy array to base64 string for frontend display"""
        try:
            # Convert back to 0-255 range
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            
            # Convert to PIL Image
            pil_img = Image.fromarray(image_array)
            
            # Convert to base64
            buffered = io.BytesIO()
            pil_img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return f"data:image/png;base64,{img_str}"
            
        except Exception as e:
            raise ValueError(f"Error converting image to base64: {str(e)}")
    
    @staticmethod
    def save_image(image_array, filename, folder):
        """Save image to file system"""
        try:
            # Convert back to 0-255 range if normalized
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            
            # Ensure image is in correct format
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            filepath = f"{folder}/{filename}"
            cv2.imwrite(filepath, image_array)
            return filepath
            
        except Exception as e:
            raise ValueError(f"Error saving image: {str(e)}")