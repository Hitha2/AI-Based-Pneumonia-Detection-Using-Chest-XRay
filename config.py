# config.py
SECRET_KEY = "replace_this_with_a_random_secret_for_production"
UPLOAD_FOLDER = "uploads"
DB_PATH = "database/pneumonia.db"
MODEL_PATH = "models/pneumonia_model.h5"
ALLOWED_EXT = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB
IMG_SIZE = (224, 224)
