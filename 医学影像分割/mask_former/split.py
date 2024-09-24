import os
import shutil
import random

def split_train_val(train_root, val_root, val_ratio=0.2):
    # Create ValDataset directory structure
    os.makedirs(os.path.join(val_root, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_root, 'masks'), exist_ok=True)

    # Get the list of image files
    image_files = os.listdir(os.path.join(train_root, 'images'))

    # Shuffle and split the image files
    random.shuffle(image_files)
    val_size = int(len(image_files) * val_ratio)
    val_images = image_files[:val_size]

    # Copy images and corresponding masks to ValDataset and then delete from TrainDataset
    for image_file in val_images:
        image_path = os.path.join(train_root, 'images', image_file)
        mask_file = image_file.split(".")[0] + ".png"
        mask_path = os.path.join(train_root, 'masks', mask_file)

        # Check if the mask exists
        if not os.path.exists(mask_path):
            print(f"Mask for image {image_file} not found, skipping.")
            continue

        # Copy image and mask to ValDataset
        shutil.copy2(image_path, os.path.join(val_root, 'images', image_file))
        shutil.copy2(mask_path, os.path.join(val_root, 'masks', mask_file))

        # Delete image and mask from TrainDataset
        os.remove(image_path)
        os.remove(mask_path)

    print(f"Copied and removed {val_size} images and masks from TrainDataset to ValDataset.")

# Usage
train_dataset_path = "data/polyp/TrainDataset"
val_dataset_path = "data/polyp/ValDataset"
split_train_val(train_dataset_path, val_dataset_path, val_ratio=0.2)
