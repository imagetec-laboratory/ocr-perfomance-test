"""Test command - OCR model performance testing"""

import json
import os
import random
import statistics
import typer
from pathlib import Path
from time import time
from typing import Optional, Dict, List
import numpy as np

import cv2
from rich.console import Console
from rich.table import Table
from side_project.models import OCREngine

console = Console()

def convert_to_serializable(obj):
    """Convert numpy arrays and other non-serializable objects to JSON-compatible formats"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return str(obj)
    else:
        return obj

def get_image_dimensions(file_path: str) -> tuple:
    """Get image dimensions (width, height)"""
    img = cv2.imread(file_path)
    if img is None:
        return (0, 0)
    height, width = img.shape[:2]
    return (width, height)

def test_command(
    input_path: str = typer.Argument(..., help="Path to image file or folder containing images"),
    models: List[str] = typer.Option(["paddleocr", "easyocr", "pytesseract"], "--models", "-m", help="OCR models to compare (can specify multiple)"),
    device: str = typer.Option("cpu", "--device", "-d", help="Device to run model on (cpu/gpu)"),
    limit_mem: Optional[float] = typer.Option(None, "--limit-mem", help="Memory limit fraction (0.0-1.0)"),
    samples: int = typer.Option(5, "--samples", "-s", help="Number of sample images per resolution category"),
    seed: Optional[int] = typer.Option(42, "--seed", help="Random seed for reproducible sampling"),
    output: str = typer.Option("results.json", "--output", "-o", help="Output file for results")
):
    """Test OCR model performance on image(s)"""

    # Validate input path
    input_path = Path(input_path)
    if not input_path.exists():
        console.print(f"❌ Error: Path '{input_path}' does not exist")
        raise typer.Exit(1)

    # Get image paths
    image_extensions = {'.jpg', '.jpeg', '.png'}
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

    console.print(f"🔍 Testing models: {', '.join(models)} on device: {device.upper()}")
    console.print("⏳ Initializing models...")
    ocr_models: Dict[str, OCREngine] = {}
    for model_name in models:
        if model_name.lower() == "paddleocr":
            try:
                from side_project.models.paddleocr import PaddleOCRModel
                ocr_models["PaddleOCR"] = PaddleOCRModel(device=device, limit_mem=limit_mem)
            except ImportError:
                console.print("[yellow]⚠️  PaddleOCR not installed. Skipping.[/yellow]")
        elif model_name.lower() == "easyocr":
            try:
                from side_project.models.easyocr import EasyOCRModel
                ocr_models["EasyOCR"] = EasyOCRModel(device=device, limit_mem=limit_mem)
            except ImportError:
                console.print("[yellow]⚠️  EasyOCR not installed. Skipping.[/yellow]")
        elif model_name.lower() == "pytesseract":
            try:
                from side_project.models.pytesseract import PyTesseractModel
                ocr_models["PyTesseract"] = PyTesseractModel(device='cpu', limit_mem=limit_mem)
            except ImportError:
                console.print("[yellow]⚠️  PyTesseract not installed. Skipping.[/yellow]")
        else:
            console.print(f"[red]❌ Unknown model '{model_name}'. Skipping.[/red]")

    console.print(f"[green]Initialized models: [{', '.join(ocr_models.keys())}][/green]")
    
    resolution_buckets = {"small": [], "medium": [], "large": []}
    for img_path in image_paths:
        width, height = get_image_dimensions(img_path)
        if width == 0 or height == 0:
            console.print(f"[yellow]⚠️  Could not read image '{img_path}'. Skipping.[/yellow]")
            continue
        if width * height < 640 * 480:
            resolution_buckets["small"].append(img_path)
        elif width * height < 1920 * 1080:
            resolution_buckets["medium"].append(img_path)
        else:
            resolution_buckets["large"].append(img_path)

    if seed is not None:
        random.seed(seed)
        console.print(f"🎲 Random seed set to: {seed}")

    sampled_images = []
    for bucket_name, bucket_images in resolution_buckets.items():
        if bucket_images:
            sample_size = min(samples, len(bucket_images))
            sampled = random.sample(bucket_images, sample_size)
            sampled_images.extend(sampled)
            console.print(f"📊 Sampled {sample_size} images from {bucket_name} resolution bucket")
            
            for img in sampled:
                console.print(f"[reset]   - {Path(img).name}[/reset] [red]({get_image_dimensions(img)[0]}x{get_image_dimensions(img)[1]})[/red]")
    
    if not sampled_images:
        console.print("❌ No images to test after sampling")
        raise typer.Exit(1)
    
    console.print(f"🎯 Total images selected for testing: {len(sampled_images)}")
    
    # Initialize results storage
    results = {
        "test_info": {
            "total_images": len(sampled_images),
            "models_tested": list(ocr_models.keys()),
            "device": device,
            "seed": seed,
            "samples_per_bucket": samples,
            "resolution_distribution": {
                "small": len(resolution_buckets["small"]),
                "medium": len(resolution_buckets["medium"]),
                "large": len(resolution_buckets["large"])
            }
        },
        "detailed_results": [],
        "summary": {},
        "summary_by_resolution": {}
    }
    
    # Test each model on all sampled images
    console.print("\n🧪 Starting OCR testing...")
    
    for model_name, model in ocr_models.items():
        console.print(f"\n📊 Testing {model_name}...")
        model_results = {
            "model": model_name,
            "processing_times": [],
            "text_lengths": [],
            "images_processed": 0,
            "images_failed": 0
        }
        
        # Track results by resolution
        resolution_results = {
            "small": {"processing_times": [], "text_lengths": [], "processed": 0, "failed": 0},
            "medium": {"processing_times": [], "text_lengths": [], "processed": 0, "failed": 0},
            "large": {"processing_times": [], "text_lengths": [], "processed": 0, "failed": 0}
        }
        
        for i, img_path in enumerate(sampled_images, 1):
            console.print(f"  Processing image {i}/{len(sampled_images)}: {Path(img_path).name}")
            
            # Determine resolution category for this image
            width, height = get_image_dimensions(img_path)
            if width * height < 640 * 480:
                resolution_category = "small"
            elif width * height < 1920 * 1080:
                resolution_category = "medium"
            else:
                resolution_category = "large"
            
            try:
                # Measure processing time
                start_time = time()
                extracted_text = model.predict(img_path)
                processing_time = time() - start_time
                
                # Convert extracted text to string if it's not already
                if extracted_text is not None:
                    extracted_text = str(extracted_text)
                else:
                    extracted_text = ""
                
                model_results["processing_times"].append(processing_time)
                model_results["text_lengths"].append(len(extracted_text))
                model_results["images_processed"] += 1
                
                # Track by resolution
                resolution_results[resolution_category]["processing_times"].append(processing_time)
                resolution_results[resolution_category]["text_lengths"].append(len(extracted_text))
                resolution_results[resolution_category]["processed"] += 1
                
                # Store detailed result
                results["detailed_results"].append({
                    "model": model_name,
                    "image": Path(img_path).name,
                    "image_path": img_path,
                    "resolution": f"{width}x{height}",
                    "resolution_category": resolution_category,
                    "processing_time": float(processing_time),
                    "text_length": len(extracted_text),
                    "extracted_text": extracted_text[:100] + "..." if len(extracted_text) > 100 else extracted_text
                })
                
            except Exception as e:
                console.print(f"    [red]❌ Failed: {str(e)}[/red]")
                model_results["images_failed"] += 1
                resolution_results[resolution_category]["failed"] += 1
        
        # Calculate model statistics
        if model_results["processing_times"]:
            avg_time = float(statistics.mean(model_results["processing_times"]))
            median_time = float(statistics.median(model_results["processing_times"]))
            min_time = float(min(model_results["processing_times"]))
            max_time = float(max(model_results["processing_times"]))
            avg_text_length = float(statistics.mean(model_results["text_lengths"]))
            
            results["summary"][model_name] = {
                "average_processing_time": avg_time,
                "median_processing_time": median_time,
                "min_processing_time": min_time,
                "max_processing_time": max_time,
                "average_text_length": avg_text_length,
                "success_rate": float(model_results["images_processed"] / len(sampled_images) * 100),
                "total_processed": int(model_results["images_processed"]),
                "total_failed": int(model_results["images_failed"])
            }
            
            # Calculate statistics by resolution
            results["summary_by_resolution"][model_name] = {}
            for res_category, res_data in resolution_results.items():
                if res_data["processing_times"]:
                    total_images_in_category = res_data["processed"] + res_data["failed"]
                    results["summary_by_resolution"][model_name][res_category] = {
                        "average_processing_time": float(statistics.mean(res_data["processing_times"])),
                        "median_processing_time": float(statistics.median(res_data["processing_times"])),
                        "min_processing_time": float(min(res_data["processing_times"])),
                        "max_processing_time": float(max(res_data["processing_times"])),
                        "average_text_length": float(statistics.mean(res_data["text_lengths"])),
                        "success_rate": float(res_data["processed"] / total_images_in_category * 100) if total_images_in_category > 0 else 0.0,
                        "total_processed": int(res_data["processed"]),
                        "total_failed": int(res_data["failed"])
                    }
                else:
                    results["summary_by_resolution"][model_name][res_category] = {
                        "average_processing_time": 0.0,
                        "median_processing_time": 0.0,
                        "min_processing_time": 0.0,
                        "max_processing_time": 0.0,
                        "average_text_length": 0.0,
                        "success_rate": 0.0,
                        "total_processed": 0,
                        "total_failed": int(res_data["failed"])
                    }
            
            console.print(f"  ✅ {model_name} completed: {model_results['images_processed']}/{len(sampled_images)} images")
    
    # Display summary table
    console.print("\n📋 Performance Summary:")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Avg Time (s)", justify="right")
    table.add_column("Median Time (s)", justify="right")
    table.add_column("Min Time (s)", justify="right")
    table.add_column("Max Time (s)", justify="right")
    
    for model_name, stats in results["summary"].items():
        table.add_row(
            model_name,
            f"{stats['average_processing_time']:.3f}",
            f"{stats['median_processing_time']:.3f}",
            f"{stats['min_processing_time']:.3f}",
            f"{stats['max_processing_time']:.3f}"
        )
    
    console.print(table)
    
    # Display summary tables by resolution
    for resolution_category in ["small", "medium", "large"]:
        console.print(f"\n📋 Performance Summary - {resolution_category.upper()} Resolution:")
        res_table = Table(show_header=True, header_style="bold magenta")
        res_table.add_column("Model", style="cyan", no_wrap=True)
        res_table.add_column("Avg Time (s)", justify="right")
        res_table.add_column("Median Time (s)", justify="right")
        res_table.add_column("Min Time (s)", justify="right")
        res_table.add_column("Max Time (s)", justify="right")
        
        has_data = False
        for model_name, res_stats in results["summary_by_resolution"].items():
            if resolution_category in res_stats and res_stats[resolution_category]["total_processed"] + res_stats[resolution_category]["total_failed"] > 0:
                has_data = True
                stats = res_stats[resolution_category]
                res_table.add_row(
                    model_name,
                    f"{stats['average_processing_time']:.3f}",
                    f"{stats['median_processing_time']:.3f}",
                    f"{stats['min_processing_time']:.3f}",
                    f"{stats['max_processing_time']:.3f}"
                )
        
        if has_data:
            console.print(res_table)
        else:
            console.print(f"  [dim]No images tested in {resolution_category} resolution category[/dim]")
    
    # Save results to file
    serializable_results = convert_to_serializable(results)
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    console.print(f"\n💾 Results saved to: {output}")
    console.print("🎉 Testing completed!")