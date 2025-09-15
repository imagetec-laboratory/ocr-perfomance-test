"""Test command - OCR model performance testing"""

import json
import os
import statistics
import typer
from pathlib import Path
from time import time
from typing import Optional, Dict, List

import cv2
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from side_project.models import OCREngine
from side_project.models.paddleocr import PaddleOCRModel

console = Console()


def get_image_dimensions(file_path: str) -> tuple:
    """Get image dimensions (width, height)"""
    img = cv2.imread(file_path)
    if img is None:
        return (0, 0)
    height, width = img.shape[:2]
    return (width, height)


def get_image_resolution_category(width: int, height: int) -> str:
    """Categorize image by resolution"""
    total_pixels = width * height

    if total_pixels < 500000:  # < 0.5MP (e.g., 640x480 = 307k pixels)
        return "low"
    elif total_pixels < 2000000:  # < 2MP (e.g., 1920x1080 = 2.07M pixels)
        return "medium"
    else:  # >= 2MP (e.g., 2048x1536 = 3.15M pixels)
        return "high"


def categorize_by_resolution(image_paths: List[str]) -> Dict[str, List[str]]:
    """Categorize images by resolution"""
    resolution_categories = {
        "low": [],      # < 0.5MP
        "medium": [],   # 0.5-2MP
        "high": []      # > 2MP
    }

    for path in image_paths:
        width, height = get_image_dimensions(path)
        if width > 0 and height > 0:
            category = get_image_resolution_category(width, height)
            resolution_categories[category].append(path)

    return resolution_categories


def test_single_image(model: OCREngine, image_path: str) -> Dict:
    """Test OCR on a single image"""
    console.print(f"Testing image: {Path(image_path).name}")
    width, height = get_image_dimensions(image_path)
    file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
    console.print(f"Resolution: {width}x{height} pixels, File size: {file_size_mb:.2f} MB")

    start_time = time()
    try:
        with console.status("[bold green]Running OCR...[/bold green]"):
            result = model.predict(image_path)

        elapsed = time() - start_time

        if result and len(result) > 0:
            # PaddleOCR returns list of [bbox, (text, confidence)]
            scores = [item[1][1] for item in result[0] if len(item) > 1 and len(item[1]) > 1] if result[0] else []
            avg_score = statistics.mean(scores) if scores else 0.0
            blocks_count = len(scores)
        else:
            avg_score = 0.0
            blocks_count = 0

        console.print(f"✅ Completed in {elapsed:.2f}s")
        console.print(f"📊 Avg score: {avg_score:.4f}, Blocks: {blocks_count}")

        return {
            "image": Path(image_path).name,
            "width": width,
            "height": height,
            "file_size_mb": file_size_mb,
            "time": elapsed,
            "avg_score": avg_score,
            "blocks": blocks_count,
            "success": True
        }

    except Exception as e:
        elapsed = time() - start_time
        console.print(f"❌ Failed: {str(e)}")
        return {
            "image": Path(image_path).name,
            "width": width,
            "height": height,
            "file_size_mb": file_size_mb,
            "time": elapsed,
            "avg_score": 0.0,
            "blocks": 0,
            "success": False,
            "error": str(e)
        }


def test_command(
    input_path: str = typer.Argument(..., help="Path to image file or folder containing images"),
    model: str = typer.Option("paddleocr", "--model", "-m", help="OCR model to use"),
    device: str = typer.Option("cpu", "--device", "-d", help="Device to run model on (cpu/gpu)"),
    limit_mem: Optional[float] = typer.Option(None, "--limit-mem", help="Memory limit fraction (0.0-1.0)"),
    output: str = typer.Option("results.json", "--output", "-o", help="Output file for results")
):
    """Test OCR model performance on image(s)"""

    # Validate input path
    input_path = Path(input_path)
    if not input_path.exists():
        console.print(f"❌ Error: Path '{input_path}' does not exist")
        raise typer.Exit(1)

    # Get image paths
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    if input_path.is_file():
        if input_path.suffix.lower() not in image_extensions:
            console.print(f"❌ Error: '{input_path}' is not a supported image file")
            raise typer.Exit(1)
        image_paths = [str(input_path)]
        console.print(f"🖼️  Testing single image: {input_path.name}")
    else:
        # Walk through folder
        image_paths = []
        for file_path in input_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_paths.append(str(file_path))

        if not image_paths:
            console.print(f"❌ Error: No supported images found in '{input_path}'")
            raise typer.Exit(1)

        console.print(f"📁 Found {len(image_paths)} images in folder")

    # Initialize model
    console.print(f"🤖 Initializing {model} model on {device}...")
    try:
        if model.lower() == "paddleocr":
            ocr_model = PaddleOCRModel(device=device, limit_mem=limit_mem)
        else:
            console.print(f"❌ Error: Model '{model}' not supported yet")
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ Error initializing model: {e}")
        raise typer.Exit(1)

    # Test images
    results = []

    if len(image_paths) == 1:
        # Single image test
        result = test_single_image(ocr_model, image_paths[0])
        results.append(result)
    else:
        # Multiple images - categorize by resolution and test
        resolution_categories = categorize_by_resolution(image_paths)

        console.print("\n📊 Image distribution by resolution:")
        console.print(f"  Low (<0.5MP): {len(resolution_categories['low'])} images")
        console.print(f"  Medium (0.5-2MP): {len(resolution_categories['medium'])} images")
        console.print(f"  High (>2MP): {len(resolution_categories['high'])} images")

        # Test each category
        for category, paths in resolution_categories.items():
            if not paths:
                continue

            console.print(f"\n🔍 Testing {category} resolution images ({len(paths)} images):")

            category_results = []
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task(f"Processing {category} resolution images...", total=len(paths))

                for path in paths:
                    result = test_single_image(ocr_model, path)
                    category_results.append(result)
                    results.append(result)
                    progress.update(task, advance=1)

            # Calculate category averages
            successful_results = [r for r in category_results if r["success"]]
            if successful_results:
                avg_time = statistics.mean([r["time"] for r in successful_results])
                avg_score = statistics.mean([r["avg_score"] for r in successful_results])
                avg_blocks = statistics.mean([r["blocks"] for r in successful_results])

                console.print(f"📈 {category.title()} resolution averages:")
                console.print(f"   Time: {avg_time:.2f}s")
                console.print(f"   Score: {avg_score:.4f}")
                console.print(f"   Blocks: {avg_blocks:.1f}")

    # Overall summary
    console.print("\n📋 Overall Summary:")
    successful_results = [r for r in results if r["success"]]
    total_images = len(results)
    successful_images = len(successful_results)

    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Images", str(total_images))
    table.add_row("Successful", str(successful_images))
    table.add_row("Failed", str(total_images - successful_images))

    if successful_results:
        avg_time = statistics.mean([r["time"] for r in successful_results])
        avg_score = statistics.mean([r["avg_score"] for r in successful_results])
        avg_blocks = statistics.mean([r["blocks"] for r in successful_results])

        table.add_row("Avg Time", f"{avg_time:.2f}s")
        table.add_row("Avg Score", f"{avg_score:.4f}")
        table.add_row("Avg Blocks", f"{avg_blocks:.1f}")

    console.print(table)

    # Save results to file
    output_path = Path(output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    console.print(f"💾 Results saved to: {output_path}")
    console.print("✨ Testing completed!")