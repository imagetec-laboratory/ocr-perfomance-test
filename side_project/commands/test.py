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


def categorize_by_resolution(image_paths: List[str], samples_per_category: int = 3) -> Dict[str, List[str]]:
    """Categorize images by resolution and limit samples per category"""
    resolution_categories = {
        "low": [],      # < 0.5MP
        "medium": [],   # 0.5-2MP
        "high": []      # > 2MP
    }

    for path in image_paths:
        width, height = get_image_dimensions(path)
        if width > 0 and height > 0:
            category = get_image_resolution_category(width, height)
            # Only add if we haven't reached the limit for this category
            if len(resolution_categories[category]) < samples_per_category:
                resolution_categories[category].append(path)
        
        # Stop if all categories are full
        if all(len(paths) >= samples_per_category for paths in resolution_categories.values()):
            break

    return resolution_categories


def initialize_models(model_names: List[str], device: str, limit_mem: Optional[float]) -> Dict[str, OCREngine]:
    """Initialize multiple OCR models"""
    models = {}
    
    for model_name in model_names:
        console.print(f"🤖 Initializing {model_name.upper()} model on {device}...")
        try:
            if model_name.lower() == "paddleocr":
                models[model_name] = PaddleOCRModel(device=device, limit_mem=limit_mem)
            elif model_name.lower() == "easyocr":
                console.print(f"⚠️  EasyOCR model is not implemented yet, skipping...")
                continue
            elif model_name.lower() == "pytesseract":
                console.print(f"⚠️  Pytesseract model is not implemented yet, skipping...")
                continue
            else:
                console.print(f"❌ Error: Model '{model_name}' not supported yet")
                continue
        except Exception as e:
            console.print(f"❌ Error initializing {model_name}: {e}")
            continue
    
    if not models:
        console.print("❌ No models were successfully initialized")
        raise typer.Exit(1)
    
    return models


def test_single_image_multi_models(models: Dict[str, OCREngine], image_path: str) -> Dict:
    """Test OCR on a single image with multiple models"""
    console.print(f"Testing image: {Path(image_path).name}")
    width, height = get_image_dimensions(image_path)
    file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
    console.print(f"Resolution: {width}x{height} pixels, File size: {file_size_mb:.2f} MB")

    results = {}
    
    for model_name, model in models.items():
        start_time = time()
        try:
            with console.status(f"[bold green]Running {model_name.upper()} OCR...[/bold green]"):
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

            console.print(f"✅ {model_name.upper()}: Completed in {elapsed:.2f}s")
            console.print(f"📊 {model_name.upper()}: Avg score: {avg_score:.4f}, Blocks: {blocks_count}")

            results[model_name] = {
                "image": Path(image_path).name,
                "model": model_name,
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
            console.print(f"❌ {model_name.upper()}: Failed: {str(e)}")
            results[model_name] = {
                "image": Path(image_path).name,
                "model": model_name,
                "width": width,
                "height": height,
                "file_size_mb": file_size_mb,
                "time": elapsed,
                "avg_score": 0.0,
                "blocks": 0,
                "success": False,
                "error": str(e)
            }
    
    return results


def test_command(
    input_path: str = typer.Argument(..., help="Path to image file or folder containing images"),
    models: List[str] = typer.Option(["paddleocr", "easyocr", "pytesseract"], "--models", "-m", help="OCR models to compare (can specify multiple)"),
    device: str = typer.Option("cpu", "--device", "-d", help="Device to run model on (cpu/gpu)"),
    limit_mem: Optional[float] = typer.Option(None, "--limit-mem", help="Memory limit fraction (0.0-1.0)"),
    samples: int = typer.Option(3, "--samples", "-s", help="Number of sample images per resolution category"),
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

    # Initialize models
    ocr_models = initialize_models(models, device, limit_mem)

    # Test images
    results = []

    if len(image_paths) == 1:
        # Single image test
        model_results = test_single_image_multi_models(ocr_models, image_paths[0])
        for model_name, result in model_results.items():
            results.append(result)
    else:
        # Multiple images - categorize by resolution and test
        resolution_categories = categorize_by_resolution(image_paths, samples_per_category=samples)

        console.print(f"\n📊 Image distribution by resolution (max {samples} samples per category):")
        total_found = sum(len(paths) for paths in resolution_categories.values())
        console.print(f"  Low (<0.5MP): {len(resolution_categories['low'])} images")
        console.print(f"  Medium (0.5-2MP): {len(resolution_categories['medium'])} images")
        console.print(f"  High (>2MP): {len(resolution_categories['high'])} images")
        console.print(f"  Total selected: {total_found} images")

        # Test each category and collect results
        category_summaries = {}
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
                    model_results = test_single_image_multi_models(ocr_models, path)
                    for model_name, result in model_results.items():
                        category_results.append(result)
                        results.append(result)
                    progress.update(task, advance=1)

            # Calculate category averages per model
            model_summaries = {}
            for model_name in ocr_models.keys():
                model_results = [r for r in category_results if r.get("model") == model_name]
                successful_results = [r for r in model_results if r["success"]]
                
                if successful_results:
                    avg_time = statistics.mean([r["time"] for r in successful_results])
                    avg_score = statistics.mean([r["avg_score"] for r in successful_results])
                    avg_blocks = statistics.mean([r["blocks"] for r in successful_results])
                    
                    model_summaries[model_name] = {
                        "count": len(successful_results),
                        "avg_time": avg_time,
                        "avg_score": avg_score,
                        "avg_blocks": avg_blocks,
                        "failed": len(model_results) - len(successful_results)
                    }
                else:
                    model_summaries[model_name] = {
                        "count": 0,
                        "avg_time": 0,
                        "avg_score": 0,
                        "avg_blocks": 0,
                        "failed": len(model_results)
                    }

            # Display immediate summary for this category
            console.print(f"\n📊 {category.title()} Resolution Summary:")
            cat_table = Table()
            cat_table.add_column("Model", style="cyan")
            cat_table.add_column("Success", style="green")
            cat_table.add_column("Failed", style="red")
            cat_table.add_column("Avg Time (s)", style="yellow")
            cat_table.add_column("Avg Score", style="magenta")
            cat_table.add_column("Avg Blocks", style="blue")
            
            for model_name, summary in model_summaries.items():
                cat_table.add_row(
                    model_name.upper(),
                    str(summary["count"]),
                    str(summary["failed"]),
                    f"{summary['avg_time']:.2f}" if summary["count"] > 0 else "N/A",
                    f"{summary['avg_score']:.4f}" if summary["count"] > 0 else "N/A",
                    f"{summary['avg_blocks']:.1f}" if summary["count"] > 0 else "N/A"
                )
                
            console.print(cat_table)

        # Display final comparison table for all categories
        console.print("\n📊 Final Model Comparison Across All Categories:")
        final_table = Table()
        final_table.add_column("Category", style="cyan")
        final_table.add_column("Model", style="blue")
        final_table.add_column("Images", style="green")
        final_table.add_column("Avg Time (s)", style="yellow")
        final_table.add_column("Avg Score", style="magenta")
        
        for category in ["low", "medium", "high"]:
            if category in [cat for cat, paths in resolution_categories.items() if paths]:
                for model_name in ocr_models.keys():
                    model_results = [r for r in results if r.get("model") == model_name and get_image_resolution_category(r["width"], r["height"]) == category]
                    successful_results = [r for r in model_results if r["success"]]
                    
                    if successful_results:
                        avg_time = statistics.mean([r["time"] for r in successful_results])
                        avg_score = statistics.mean([r["avg_score"] for r in successful_results])
                        
                        final_table.add_row(
                            f"{category.title()}" if model_name == list(ocr_models.keys())[0] else "",
                            model_name.upper(),
                            str(len(successful_results)),
                            f"{avg_time:.2f}",
                            f"{avg_score:.4f}"
                        )
        
        console.print(final_table)

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