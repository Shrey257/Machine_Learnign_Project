import os
import yaml
import argparse
from ultralytics import YOLO
import torch
import random
import numpy as np

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def create_dataset_yaml(dataset_path, output_path):
    """Create YAML file for dataset configuration"""
    data = {
        'path': dataset_path,
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {
            0: 'player',
            1: 'ball'
        },
        'nc': 2  # number of classes
    }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write YAML file
    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"Dataset configuration saved to {output_path}")
    return output_path

def train_model(dataset_path, epochs=100, batch_size=16, img_size=1280, pretrained_weights=None):
    """Train YOLOv8 model for football analysis"""
    # Set seed for reproducibility
    set_seed(42)
    
    # Create dataset YAML file
    dataset_yaml = create_dataset_yaml(dataset_path, os.path.join(dataset_path, 'dataset.yaml'))
    
    # Initialize YOLOv8 model
    if pretrained_weights:
        print(f"Loading pretrained weights from {pretrained_weights}")
        model = YOLO(pretrained_weights)
    else:
        print("Using YOLOv8x model (largest variant for maximum accuracy)")
        model = YOLO('yolov8x.pt')  # Use YOLOv8x for maximum accuracy
    
    # Define output directory
    output_dir = os.path.join('models', 'football-players-detection')
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for GPU availability
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")
    
    # Train the model with intensive data augmentation
    model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        workers=8,
        patience=20,
        project=output_dir,
        name='training',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',  # Using AdamW optimizer
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0001,
        warmup_epochs=3,
        warmup_momentum=0.8,
        close_mosaic=10,  # Disable mosaic in last 10 epochs
        cos_lr=True,      # Use cosine learning rate
        # Data augmentation settings
        augment=True,
        mixup=0.1,
        degrees=10.0,      # Rotation
        translate=0.2,     # Translation
        scale=0.2,         # Scale
        shear=0.0,         # Shear
        perspective=0.0,   # Perspective
        flipud=0.0,        # Flip up-down
        fliplr=0.5,        # Flip left-right
        mosaic=1.0,        # Mosaic
        copy_paste=0.1,    # Copy-paste
        auto_augment='randaugment'  # Auto augmentation
    )
    
    # Validate the model
    metrics = model.val()
    print(f"Validation metrics: {metrics}")
    
    # Save model
    model_path = os.path.join(output_dir, 'weights', 'best.pt')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Copy best model to final location
    best_model_path = os.path.join(output_dir, 'training', 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        import shutil
        shutil.copy(best_model_path, model_path)
        print(f"Best model saved to {model_path}")
    else:
        print(f"Warning: Best model not found at {best_model_path}")
    
    return model_path

def create_test_time_augmentation_ensemble(model_path, output_path=None):
    """Create a TTA (Test Time Augmentation) ensemble model for higher accuracy"""
    print("Creating Test Time Augmentation ensemble...")
    
    if output_path is None:
        output_path = model_path.replace('.pt', '_tta.pt')
    
    # Load the trained model
    model = YOLO(model_path)
    
    # Export the model with TTA enabled
    model.export(format='onnx', imgsz=1280, dynamic=True, opset=12)
    
    print(f"TTA model exported to {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 for football analysis')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=1280, help='Image size')
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pretrained weights')
    parser.add_argument('--tta', action='store_true', help='Create Test Time Augmentation ensemble')
    args = parser.parse_args()
    
    print("=== Football Analysis Model Training ===")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.img_size}")
    print(f"Pretrained weights: {args.pretrained}")
    
    # Train the model
    model_path = train_model(
        args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        pretrained_weights=args.pretrained
    )
    
    # Create TTA ensemble if requested
    if args.tta:
        create_test_time_augmentation_ensemble(model_path)
    
    print("Training complete!")

if __name__ == "__main__":
    main() 