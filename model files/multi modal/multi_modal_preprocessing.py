import os
import cv2
import pandas as pd
import mediapipe as mp

# Paths
image_dataset_path = r"C:\path\to\image_dataset"  # Directory containing subfolders for each ASL class
output_csv_path = r"C:\path\to\multi_modal_dataset.csv"

# Mediapipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

# Initialize a list to store skeletal features, image paths, and labels
dataset = []

# Loop through each class folder (e.g., A, B, C...)
for class_label in os.listdir(image_dataset_path):
    class_folder = os.path.join(image_dataset_path, class_label)
    if not os.path.isdir(class_folder):
        continue

    print(f"Processing class: {class_label}")
    # Loop through each image in the class folder
    for image_file in os.listdir(class_folder):
        image_path = os.path.join(class_folder, image_file)

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read {image_path}")
            continue

        # Convert the image to RGB (required by Mediapipe)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image with Mediapipe Hands
        result = hands.process(rgb_image)

        if result.multi_hand_landmarks:
            # Extract the first hand's landmarks
            landmarks = result.multi_hand_landmarks[0]
            # Flatten the 21 landmarks (x, y) into a single vector
            flattened_landmarks = []
            for lm in landmarks.landmark:
                flattened_landmarks.extend([lm.x, lm.y])  # Only x, y (ignore z)
            
            # Append the skeletal features, image path, and label to the dataset
            dataset.append(flattened_landmarks + [image_path, class_label])
        else:
            print(f"No hand detected in {image_path}")

# Define column names
landmark_columns = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)]
columns = landmark_columns + ["image_path", "label"]

# Convert the dataset into a DataFrame
df = pd.DataFrame(dataset, columns=columns)

# Save the dataset as a CSV file
df.to_csv(output_csv_path, index=False)
print(f"Multi-modal dataset saved as CSV at {output_csv_path}")
