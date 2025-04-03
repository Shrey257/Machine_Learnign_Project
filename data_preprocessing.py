import os
import argparse
import random
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split

def ensure_dirs(base_path):
    """Create necessary directory structure for training"""
    dirs = [
        os.path.join(base_path, 'train', 'images'),
        os.path.join(base_path, 'train', 'labels'),
        os.path.join(base_path, 'val', 'images'),
        os.path.join(base_path, 'val', 'labels'),
        os.path.join(base_path, 'test', 'images'),
        os.path.join(base_path, 'test', 'labels')
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    return dirs

def convert_annotations(input_annotations, image_width, image_height, class_mapping=None):
    """
    Convert annotations to YOLO format (normalized coordinates)
    
    Expected input format: [class_id, x1, y1, x2, y2] or [class_name, x1, y1, x2, y2]
    Output format: [class_id, x_center, y_center, width, height] (normalized)
    """
    yolo_annotations = []
    
    for ann in input_annotations:
        if len(ann) != 5:
            continue
        
        # Parse annotation
        if isinstance(ann[0], str) and class_mapping:
            # If class is provided as string, map to integer
            if ann[0] in class_mapping:
                class_id = class_mapping[ann[0]]
            else:
                print(f"Warning: Unknown class '{ann[0]}'. Skipping annotation.")
                continue
        else:
            # Assume class is already an integer
            class_id = int(ann[0])
        
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(float, ann[1:])
        
        # Calculate normalized center coordinates and dimensions
        x_center = (x1 + x2) / 2 / image_width
        y_center = (y1 + y2) / 2 / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height
        
        # Ensure values are within 0-1 range
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        # Add to YOLO annotations
        yolo_annotations.append([class_id, x_center, y_center, width, height])
    
    return yolo_annotations

def split_dataset(image_paths, split_ratio=(0.8, 0.1, 0.1), seed=42):
    """Split dataset into train, validation and test sets"""
    # Ensure split ratios sum to 1
    assert sum(split_ratio) == 1.0, "Split ratios must sum to 1.0"
    
    # First split train from rest
    train_size = split_ratio[0]
    train_paths, temp_paths = train_test_split(
        image_paths, train_size=train_size, random_state=seed
    )
    
    # Then split the remaining into val and test
    val_ratio = split_ratio[1] / (split_ratio[1] + split_ratio[2])
    val_paths, test_paths = train_test_split(
        temp_paths, train_size=val_ratio, random_state=seed
    )
    
    return train_paths, val_paths, test_paths

def process_dataset(input_dir, output_dir, split_ratio=(0.8, 0.1, 0.1), seed=42):
    """Process dataset and organize into train/val/test splits"""
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Create directory structure
    ensure_dirs(output_dir)
    
    # Find all image files
    image_extensions = ['jpg', 'jpeg', 'png']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(glob(os.path.join(input_dir, f'**/*.{ext}'), recursive=True))
    
    print(f"Found {len(image_paths)} images")
    
    # Split dataset
    train_paths, val_paths, test_paths = split_dataset(image_paths, split_ratio, seed)
    
    print(f"Train: {len(train_paths)}, Validation: {len(val_paths)}, Test: {len(test_paths)}")
    
    # Process each split
    for split_name, paths in [('train', train_paths), ('val', val_paths), ('test', test_paths)]:
        process_split(split_name, paths, output_dir)

def process_split(split_name, image_paths, output_dir):
    """Process a single dataset split"""
    images_dir = os.path.join(output_dir, split_name, 'images')
    labels_dir = os.path.join(output_dir, split_name, 'labels')
    
    print(f"Processing {split_name} split...")
    
    for img_path in tqdm(image_paths):
        # Get the filename without extension
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        
        # Check if annotation file exists
        ann_path = img_path.replace(ext, '.txt')
        if not os.path.exists(ann_path):
            # Also try looking for it in a 'labels' directory
            ann_dir = os.path.dirname(img_path).replace('images', 'labels')
            ann_path = os.path.join(ann_dir, name + '.txt')
        
        if not os.path.exists(ann_path):
            print(f"Warning: No annotation found for {img_path}")
            continue
        
        # Copy image to output directory
        dst_img_path = os.path.join(images_dir, filename)
        shutil.copy(img_path, dst_img_path)
        
        # Copy annotation to output directory
        dst_ann_path = os.path.join(labels_dir, name + '.txt')
        shutil.copy(ann_path, dst_ann_path)

def apply_augmentations(input_dir, output_dir, augmentation_factor=2):
    """Apply data augmentations to increase dataset size"""
    # Find all training images
    train_images_dir = os.path.join(input_dir, 'train', 'images')
    train_labels_dir = os.path.join(input_dir, 'train', 'labels')
    
    image_paths = glob(os.path.join(train_images_dir, '*.jpg')) + \
                  glob(os.path.join(train_images_dir, '*.jpeg')) + \
                  glob(os.path.join(train_images_dir, '*.png'))
    
    print(f"Applying augmentations to {len(image_paths)} training images...")
    
    # Define augmentation operations
    augmentations = [
        ('flip_horizontal', lambda img: cv2.flip(img, 1)),
        ('brightness', lambda img: adjust_brightness(img, factor=0.8)),
        ('contrast', lambda img: adjust_contrast(img, factor=1.2)),
        ('rotate5', lambda img: rotate_image(img, angle=5)),
        ('rotate355', lambda img: rotate_image(img, angle=355)),
    ]
    
    for img_path in tqdm(image_paths):
        # Get the filename without extension
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        
        # Load the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
        
        # Get the annotation path
        ann_path = os.path.join(train_labels_dir, name + '.txt')
        if not os.path.exists(ann_path):
            print(f"Warning: No annotation found for {img_path}")
            continue
        
        # Load annotations
        with open(ann_path, 'r') as f:
            annotations = [line.strip().split() for line in f.readlines()]
        
        # Convert string values to float
        annotations = [[int(ann[0])] + [float(x) for x in ann[1:]] for ann in annotations]
        
        # For each image, apply a random subset of augmentations
        num_augmentations = min(len(augmentations), augmentation_factor)
        selected_augmentations = random.sample(augmentations, num_augmentations)
        
        for aug_name, aug_func in selected_augmentations:
            # Create new filename for augmented image
            aug_filename = f"{name}_{aug_name}{ext}"
            aug_img_path = os.path.join(train_images_dir, aug_filename)
            
            # Apply augmentation to image
            aug_img = aug_func(img.copy())
            
            # Save augmented image
            cv2.imwrite(aug_img_path, aug_img)
            
            # Create new annotations file for flipped image
            aug_ann_path = os.path.join(train_labels_dir, f"{name}_{aug_name}.txt")
            
            # Adjust annotations based on augmentation type
            if aug_name == 'flip_horizontal':
                # For horizontal flip, adjust x coordinates
                with open(aug_ann_path, 'w') as f:
                    for ann in annotations:
                        class_id, x_center, y_center, width, height = ann
                        # Flip x_center (1 - x)
                        x_center = 1.0 - x_center
                        f.write(f"{int(class_id)} {x_center} {y_center} {width} {height}\n")
            
            elif aug_name.startswith('rotate'):
                # For rotation, we would need complex transformation of bbox
                # For small angles, original annotations might still be acceptable
                with open(aug_ann_path, 'w') as f:
                    for ann in annotations:
                        class_id, x_center, y_center, width, height = ann
                        f.write(f"{int(class_id)} {x_center} {y_center} {width} {height}\n")
            
            else:
                # For brightness/contrast changes, annotations remain the same
                with open(aug_ann_path, 'w') as f:
                    for ann in annotations:
                        class_id, x_center, y_center, width, height = ann
                        f.write(f"{int(class_id)} {x_center} {y_center} {width} {height}\n")

def adjust_brightness(img, factor=0.5):
    """Adjust image brightness"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def adjust_contrast(img, factor=1.5):
    """Adjust image contrast"""
    mean = np.mean(img, axis=(0, 1))
    return np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)

def rotate_image(img, angle=10):
    """Rotate image by a small angle"""
    height, width = img.shape[:2]
    center = (width/2, height/2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply rotation
    rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
    
    return rotated_img

def download_additional_data(output_dir, num_samples=100):
    """
    Download additional football images from open source datasets
    This is a placeholder function - in a real implementation,
    you would download from actual datasets
    """
    print("Note: This is a placeholder for downloading additional data.")
    print("In a real implementation, you would connect to APIs for datasets like:")
    print("- Open Images Dataset")
    print("- COCO Dataset (filtered for sports images)")
    print("- Sports-1M dataset")
    print("- Custom football datasets from papers")
    
    # Here you would implement actual download logic
    
    print(f"To download actual data, consider:")
    print("1. Kaggle Datasets (https://www.kaggle.com/datasets)")
    print("2. Roboflow Universe (https://universe.roboflow.com/)")
    print("3. Google's Open Images Dataset (https://storage.googleapis.com/openimages/web/index.html)")

def balance_classes(dataset_dir):
    """Balance classes by adjusting sampling weights"""
    print("Analyzing class distribution...")
    
    # Count instances of each class
    class_counts = {}
    
    for split in ['train', 'val', 'test']:
        labels_dir = os.path.join(dataset_dir, split, 'labels')
        label_files = glob(os.path.join(labels_dir, '*.txt'))
        
        for label_file in label_files:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        if class_id not in class_counts:
                            class_counts[class_id] = 0
                        class_counts[class_id] += 1
    
    print("Class distribution:")
    for class_id, count in sorted(class_counts.items()):
        print(f"Class {class_id}: {count} instances")
    
    # Calculate class weights for training
    total_instances = sum(class_counts.values())
    class_weights = {}
    
    for class_id, count in class_counts.items():
        class_weights[class_id] = total_instances / (len(class_counts) * count)
    
    print("\nRecommended class weights for training:")
    for class_id, weight in sorted(class_weights.items()):
        print(f"Class {class_id}: {weight:.4f}")
    
    # Save class weights to file
    weights_path = os.path.join(dataset_dir, 'class_weights.txt')
    with open(weights_path, 'w') as f:
        for class_id, weight in sorted(class_weights.items()):
            f.write(f"{class_id} {weight:.6f}\n")
    
    print(f"\nClass weights saved to {weights_path}")
    
    return class_weights

def main():
    parser = argparse.ArgumentParser(description='Preprocess dataset for football analysis')
    parser.add_argument('--input', type=str, required=True, help='Input data directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory for processed dataset')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Training data ratio')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation data ratio')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='Test data ratio')
    parser.add_argument('--augment', action='store_true', help='Apply data augmentation')
    parser.add_argument('--augment-factor', type=int, default=2, help='Augmentation factor')
    parser.add_argument('--balance', action='store_true', help='Balance classes')
    parser.add_argument('--download-extra', action='store_true', help='Download additional data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Validate split ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-10:
        parser.error("Split ratios must sum to 1.0")
    
    # Process the dataset
    split_ratio = (args.train_ratio, args.val_ratio, args.test_ratio)
    process_dataset(args.input, args.output, split_ratio, args.seed)
    
    # Apply data augmentation if requested
    if args.augment:
        apply_augmentations(args.output, args.output, args.augment_factor)
    
    # Download additional data if requested
    if args.download_extra:
        download_additional_data(args.output)
    
    # Balance classes if requested
    if args.balance:
        balance_classes(args.output)
    
    print("Data preprocessing complete!")

if __name__ == "__main__":
    main() 