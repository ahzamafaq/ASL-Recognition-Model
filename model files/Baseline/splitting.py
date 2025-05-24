import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
processed_path = "C:/Users/akshi/Desktop/STS/asl_alphabet_processed"
split_base_path = "C:/Users/akshi/Desktop/STS/asl_alphabet_splits"

# Folders for splits
train_path = os.path.join(split_base_path, "train")
val_path = os.path.join(split_base_path, "val")
test_path = os.path.join(split_base_path, "test")

# Function to split the dataset
def split_dataset():
    if not os.path.exists(split_base_path):
        os.makedirs(split_base_path)

    for split_path in [train_path, val_path, test_path]:
        if not os.path.exists(split_path):
            os.makedirs(split_path)

    for label in os.listdir(processed_path):
        label_path = os.path.join(processed_path, label)
        if not os.path.isdir(label_path):
            continue  # Skip non-folder files

        # Create subfolders for each label in the splits
        for split_path in [train_path, val_path, test_path]:
            os.makedirs(os.path.join(split_path, label), exist_ok=True)

        # Get all files for the label
        files = [os.path.join(label_path, f) for f in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, f))]

        # Split files into train, validation, and test sets
        train_files, temp_files = train_test_split(files, test_size=0.3, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

        # Move files to their respective folders
        for file in train_files:
            shutil.copy(file, os.path.join(train_path, label))
        for file in val_files:
            shutil.copy(file, os.path.join(val_path, label))
        for file in test_files:
            shutil.copy(file, os.path.join(test_path, label))

# Run the split
split_dataset()
print("Dataset successfully split into training, validation, and test sets!")
