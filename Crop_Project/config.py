import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'crop-classification-secret-key'
    UPLOAD_FOLDER = 'static/uploads'
    RESULT_FOLDER = 'static/results'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'tif'}
    
    # Model paths
    MODEL_PATH = 'models/crop_classifier.pth'
    
    @staticmethod
    def init_app(app):
        # Create necessary directories
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.RESULT_FOLDER, exist_ok=True)