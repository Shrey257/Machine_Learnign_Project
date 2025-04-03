#!/usr/bin/env python3
import argparse
import os
import sys
import json
import time
from ultralytics import YOLO
import config
from main import run_analysis
from data_preprocessing import process_dataset, apply_augmentations, balance_classes
from train_model import train_model, create_test_time_augmentation_ensemble

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Football Analysis System CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Analyze video command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze football video")
    analyze_parser.add_argument("--input", "-i", required=True, help="Path to input video")
    analyze_parser.add_argument("--output", "-o", default=None, help="Path to output video")
    analyze_parser.add_argument("--model", "-m", default=None, help="Path to YOLOv8 model")
    analyze_parser.add_argument("--report", "-r", action="store_true", help="Generate HTML report")
    analyze_parser.add_argument("--report-path", default=None, help="Path to HTML report")
    
    # Train model command
    train_parser = subparsers.add_parser("train", help="Train YOLOv8 model")
    train_parser.add_argument("--dataset", "-d", required=True, help="Path to dataset directory")
    train_parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of training epochs")
    train_parser.add_argument("--batch-size", "-b", type=int, default=16, help="Batch size")
    train_parser.add_argument("--img-size", "-s", type=int, default=1280, help="Image size")
    train_parser.add_argument("--pretrained", "-p", default=None, help="Path to pretrained weights")
    train_parser.add_argument("--tta", action="store_true", help="Create Test Time Augmentation ensemble")
    
    # Preprocess dataset command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess dataset")
    preprocess_parser.add_argument("--input", "-i", required=True, help="Input data directory")
    preprocess_parser.add_argument("--output", "-o", required=True, help="Output directory for processed dataset")
    preprocess_parser.add_argument("--train-ratio", type=float, default=0.8, help="Training data ratio")
    preprocess_parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation data ratio")
    preprocess_parser.add_argument("--test-ratio", type=float, default=0.1, help="Test data ratio")
    preprocess_parser.add_argument("--augment", "-a", action="store_true", help="Apply data augmentation")
    preprocess_parser.add_argument("--augment-factor", type=int, default=2, help="Augmentation factor")
    preprocess_parser.add_argument("--balance", "-b", action="store_true", help="Balance classes")
    
    # List models command
    list_models_parser = subparsers.add_parser("list-models", help="List available models")
    
    return parser.parse_args()

def analyze_video(args):
    """Run analysis on a football video"""
    print(f"Analyzing video: {args.input}")
    
    # Determine output path if not specified
    if args.output is None:
        output_dir = os.path.join("output", "videos")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time())
        args.output = os.path.join(output_dir, f"analyzed_video_{timestamp}.mp4")
    
    # Determine model path if not specified
    model_path = args.model if args.model else config.MODEL_PATH
    
    # Determine report path if generating report
    report_path = args.report_path
    if args.report and report_path is None:
        report_dir = os.path.join("output", "reports")
        os.makedirs(report_dir, exist_ok=True)
        timestamp = int(time.time())
        report_path = os.path.join(report_dir, f"report_{timestamp}.html")
    
    # Run analysis
    try:
        start_time = time.time()
        result = run_analysis(args.input, args.output, model_path, args.report, report_path)
        end_time = time.time()
        
        # Print summary
        print("\nAnalysis completed successfully!")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print(f"Output video: {args.output}")
        
        if args.report:
            print(f"HTML Report: {report_path}")
        
        # Print quick stats if available
        if result and isinstance(result, dict):
            team_a_count = sum(1 for stats in result.values() if stats["team"] == "team_a")
            team_b_count = sum(1 for stats in result.values() if stats["team"] == "team_b")
            
            print("\nQuick Statistics:")
            print(f"Players Tracked: {len(result)}")
            print(f"Team A Players: {team_a_count}")
            print(f"Team B Players: {team_b_count}")
        
        return True
    except Exception as e:
        print(f"Error analyzing video: {e}")
        return False

def train_model_command(args):
    """Train YOLOv8 model with specified parameters"""
    print(f"Training model with dataset: {args.dataset}")
    
    try:
        start_time = time.time()
        model_path = train_model(
            args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            pretrained_weights=args.pretrained
        )
        end_time = time.time()
        
        print("\nTraining completed successfully!")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print(f"Model saved to: {model_path}")
        
        # Create TTA ensemble if requested
        if args.tta:
            print("\nCreating Test Time Augmentation ensemble...")
            tta_path = create_test_time_augmentation_ensemble(model_path)
            print(f"TTA model saved to: {tta_path}")
        
        return True
    except Exception as e:
        print(f"Error training model: {e}")
        return False

def preprocess_dataset_command(args):
    """Preprocess dataset with specified parameters"""
    print(f"Preprocessing dataset: {args.input} -> {args.output}")
    
    try:
        # Validate split ratios
        total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
        if abs(total_ratio - 1.0) > 1e-10:
            print("Error: Split ratios must sum to 1.0")
            return False
        
        start_time = time.time()
        
        # Process the dataset
        split_ratio = (args.train_ratio, args.val_ratio, args.test_ratio)
        process_dataset(args.input, args.output, split_ratio)
        
        # Apply data augmentation if requested
        if args.augment:
            apply_augmentations(args.output, args.output, args.augment_factor)
        
        # Balance classes if requested
        if args.balance:
            balance_classes(args.output)
        
        end_time = time.time()
        
        print("\nPreprocessing completed successfully!")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print(f"Processed dataset saved to: {args.output}")
        
        return True
    except Exception as e:
        print(f"Error preprocessing dataset: {e}")
        return False

def list_models_command():
    """List available models in the models directory"""
    models_dir = "models"
    
    if not os.path.exists(models_dir):
        print("No models directory found.")
        return
    
    models = []
    
    # Walk through the models directory
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.endswith(".pt") or file.endswith(".onnx"):
                model_path = os.path.join(root, file)
                model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
                models.append({
                    "path": model_path,
                    "size": f"{model_size_mb:.2f} MB",
                    "modified": time.ctime(os.path.getmtime(model_path))
                })
    
    if not models:
        print("No models found.")
        return
    
    # Print models
    print(f"Found {len(models)} models:")
    print("=" * 80)
    print(f"{'Path':<50} {'Size':<10} {'Last Modified':<20}")
    print("-" * 80)
    
    for model in models:
        print(f"{model['path']:<50} {model['size']:<10} {model['modified']:<20}")
    
    print("=" * 80)

def main():
    """Main CLI function"""
    args = parse_args()
    
    if args.command == "analyze":
        analyze_video(args)
    elif args.command == "train":
        train_model_command(args)
    elif args.command == "preprocess":
        preprocess_dataset_command(args)
    elif args.command == "list-models":
        list_models_command()
    else:
        print("Please specify a command: analyze, train, preprocess, or list-models")
        print("Use -h or --help for more information")

if __name__ == "__main__":
    main() 