import os
import pandas as pd
from PIL import Image

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    import whisper
except ImportError:
    whisper = None


class MediaProcessor:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.whisper_model = None

    def _is_valid_file(self, full_path: str) -> bool:
        """Validates that the file exists and is not a directory."""
        if not full_path or not isinstance(full_path, str):
            return False
        return os.path.exists(full_path) and os.path.isfile(full_path)

    def process_image(self, relative_path: str) -> str:
        """Extracts text content from image files using OCR."""
        if not relative_path or pd.isna(relative_path) or str(relative_path).strip() in [".", ""]:
            return ""

        full_path = os.path.join(self.dataset_dir, str(relative_path).strip())

        if not self._is_valid_file(full_path):
            return ""

        if pytesseract is not None:
            try:
                img = Image.open(full_path)
                text = pytesseract.image_to_string(img)
                return text.strip()
            except Exception:
                pass

        return "[Image Attachment]"

    def process_voice_note(self, relative_path: str) -> str:
        """Transcribes audio voice notes into text."""
        if not relative_path or pd.isna(relative_path) or str(relative_path).strip() in [".", ""]:
            return ""

        full_path = os.path.join(self.dataset_dir, str(relative_path).strip())

        if not self._is_valid_file(full_path):
            return ""

        if whisper is not None:
            try:
                if self.whisper_model is None:
                    self.whisper_model = whisper.load_model("tiny")
                result = self.whisper_model.transcribe(full_path)
                return result.get("text", "").strip()
            except Exception:
                pass

        return "[Voice Note Recording]"