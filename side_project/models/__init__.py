from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class OCREngine(ABC):
    """Abstract base class for OCR engines."""

    @abstractmethod
    def __init__(self, device: str = 'cpu', limit_mem: Optional[float] = None):
        """Initialize the OCR engine.

        Args:
            device (str): The device to run the model on. Defaults to 'cpu'.
            limit_mem (Optional[float]): The fraction of memory to use (0.0 to 1.0). Defaults to None.
        """
        pass

    @abstractmethod
    def predict(self, image_path: str) -> List[Dict[str, Any]]:
        """Perform OCR on the given image.

        Args:
            image_path (str): Path to the image file to process.

        Returns:
            List[Dict[str, Any]]: OCR results containing detected text and metadata.
        """
        pass