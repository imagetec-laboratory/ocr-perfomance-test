from time import time
from paddleocr import PaddleOCR  # Application layer
import paddle                    # Framework layer
from pathlib import Path
import os
from rich.console import Console
from rich import spinner
import json

console = Console()

class PaddleOCRModel:
    
    def __init__(self, device='cpu', limit_mem=None):
        """Initialize the PaddleOCRModel.

        Args:
            device (str, optional): The device to run the model on. Defaults to 'cpu'.
            limit_mem (float, optional): The fraction of CPU memory to use (0.0 to 1.0). Defaults to None.
        """
        paddle.set_device(device)
        if limit_mem:
            import psutil
            ram_info = psutil.virtual_memory()
            console.print(f"Total RAM: [bold]{ram_info.total / (1024**3):.2f} GB[/bold]")
            console.print(f"Available RAM: [bold]{ram_info.available / (1024**3):.2f} GB[/bold]")
            console.print(f"Used RAM: [bold]{ram_info.used / (1024**3):.2f} GB[/bold]")
            paddle.set_flags({"FLAGS_fraction_of_cpu_memory_to_use": limit_mem})
            allocated_for_paddle = ram_info.total * limit_mem
            console.print(f"PaddlePaddle allocated: [bold]{allocated_for_paddle / (1024**3):.2f} GB[/bold]")
            

        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_recognition_model_dir=str(pre_path / 'en_PP-OCRv5_mobile_rec'),
            text_recognition_model_name='en_PP-OCRv5_mobile_rec',
            text_detection_model_dir=str(pre_path / 'PP-OCRv5_server_det'),
            text_detection_model_name='PP-OCRv5_server_det',
        )

    def predict(self, image_path):
        return self.ocr.predict(input=image_path)

if __name__ == "__main__":
    print("Running PaddleOCRModel test...")
    model = PaddleOCRModel(limit_mem=0.1)
    with console.status("[bold green]Processing image...", spinner="dots") as status:
        image_path = "../images/img-01001-00001.jpg"
        console.print(f"[blue]Info:[/blue] Size of image: {os.path.getsize(image_path) / 1024:.2f} KB")
        console.print(f"Processing image: {image_path}")
        result = model.predict(image_path=image_path)
        console.print(f"[green]✓[/green] OCR processing completed")
        console.print(f"Result: {json.dumps(result, indent=2)}")
        for res in result:
            status.update("Saving results...")
            res.save_to_img("pretrained_models/output")
        console.print("[green]✓[/green] Processing completed")