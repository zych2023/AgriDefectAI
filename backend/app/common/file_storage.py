import os
import uuid

from fastapi import UploadFile

from app.core.config import settings

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "bmp", "webp"}


def validate_image(filename: str) -> bool:
    """Check if the file extension is in ALLOWED_EXTENSIONS."""
    ext = os.path.splitext(filename)[1].lstrip(".").lower()
    return ext in ALLOWED_EXTENSIONS


async def save_upload(file: UploadFile, sub_dir: str = "") -> str:
    """Save uploaded file with a UUID filename and return the relative path.

    Args:
        file: The uploaded file (FastAPI UploadFile).
        sub_dir: Optional subdirectory under UPLOAD_DIR (e.g. "diseases").

    Returns:
        Relative path string (e.g. "uploads/diseases/<uuid>.jpg").
    """
    ext = os.path.splitext(file.filename or "image.jpg")[1] or ".jpg"
    filename = f"{uuid.uuid4().hex}{ext}"
    target_dir = os.path.join(settings.UPLOAD_DIR, sub_dir) if sub_dir else settings.UPLOAD_DIR
    os.makedirs(target_dir, exist_ok=True)
    filepath = os.path.join(target_dir, filename)
    content = await file.read()
    with open(filepath, "wb") as f:
        f.write(content)
    return os.path.join(settings.UPLOAD_DIR, sub_dir, filename).replace("\\", "/")
