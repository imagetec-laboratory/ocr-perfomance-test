from . import OCREngine

class EasyOCRModel(OCREngine):
    def __init__(self, device='cpu', limit_mem=None):
        """Initialize the EasyOCRModel.
        Args:
            device (str, optional): The device to run the model on. Defaults to 'cpu'.
            limit_mem (float, optional): The fraction of CPU memory to use (0.0 to 1.0). Defaults to None.
        """
        import easyocr
        print(f"[blue]Initializing EasyOCRModel on {device}[/blue]")
        self.ocr = easyocr.Reader(
            lang_list=['en'],
            gpu=(device=='cuda'),
            
        )
        
    def predict(self, image_path):
        """Extract text bounding boxes from image using EasyOCR
        
        Returns:
            List of bounding boxes in format [x_min, y_min, x_max, y_max]
        """
        results = self.ocr.readtext(image_path)
        
        bboxes = []
        for (bbox, text, prob) in results:
            if prob > 0.3:  # confidence threshold
                x_min = int(min([point[0] for point in bbox]))
                y_min = int(min([point[1] for point in bbox]))
                x_max = int(max([point[0] for point in bbox]))
                y_max = int(max([point[1] for point in bbox]))
                
                bboxes.append([x_min, y_min, x_max, y_max])
        
        return bboxes
    
if __name__ == "__main__":
    import time
    from PIL import Image, ImageDraw
    from rich.console import Console

    console = Console()
    console.print("[blue]Running EasyOCRModel test...[/blue]")

    model = EasyOCRModel()
    image_path = "../images/img-01001-00001.jpg"
    start = time.time()

    console.print("[blue]Testing EasyOCRModel[/blue]")
    results = model.predict(image_path)
    console.print(f"[blue]Found {len(results)} bounding boxes[/blue]")

    # Draw boxes on image
    # new_image = Image.open(image_path).convert("RGB")
    # draw = ImageDraw.Draw(new_image)

    # for i, box in enumerate(results):
        # draw.rectangle(box, outline="red", width=2)

    # new_image.show()
    console.print(f"[green]Processing time: {time.time() - start:.2f} seconds[/green]")