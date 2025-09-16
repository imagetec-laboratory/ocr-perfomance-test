from . import OCREngine

class PyTesseractModel(OCREngine):
    def __init__(self, device='cpu', limit_mem=None):
        """Initialize the TesseractModel.
        Args:
            device (str, optional): The device to run the model on. Defaults to 'cpu'.
            limit_mem (float, optional): The fraction of CPU memory to use (0.0 to 1.0). Defaults to None.
        """
        if device != 'cpu':
            print("[yellow]Warning: Tesseract only runs on CPU. Ignoring device parameter.[/yellow]")

        import pytesseract
        self.ocr = pytesseract
        
    def predict(self, image_path):
        """Extract text bounding boxes from image using Tesseract
        
        Returns:
            List of bounding boxes in format [x_min, y_min, x_max, y_max]
        """
        from PIL import Image
        
        # Open image and get dimensions
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        # Method 1: Use image_to_data for better results
        data = self.ocr.image_to_data(img, output_type=self.ocr.Output.DICT)
        
        bboxes = []
        n_boxes = len(data['level'])
        
        for i in range(n_boxes):
            # Filter out empty detections
            if int(data['conf'][i]) > 30:  # confidence threshold
                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]
                
                # Convert to [x_min, y_min, x_max, y_max] format
                bbox = [x, y, x + w, y + h]
                bboxes.append(bbox)
        
        return bboxes

if __name__ == "__main__":
    import time
    from PIL import Image, ImageDraw
    from rich.console import Console

    console = Console()

    model = PyTesseractModel()
    image_path = "../images/img-01001-00001.jpg"
    start = time.time()

    console.print("[blue]Testing Method 1: image_to_data (recommended)[/blue]")
    results = model.predict(image_path)
    console.print(f"[blue]Found {len(results)} bounding boxes[/blue]")

    # Draw boxes on image
    new_image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(new_image)
    
    for i, box in enumerate(results):
        draw.rectangle(box, outline="red", width=2)

    console.print(f"[green]Processing time: {time.time() - start:.2f} seconds[/green]")