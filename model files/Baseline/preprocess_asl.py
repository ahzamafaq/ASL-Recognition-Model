import os
import cv2
from tensorflow.keras.preprocessingz.image import ImageDataGenerator

# Paths
dataset_path = "C:/Users/akshi/Desktop/STS/asl_alphabet_train/asl_alphabet_train"  # Training dataset path
processed_path = "C:/Users/akshi/Desktop/STS/asl_alphabet_processed"  # Path to save processed images

# Parameters
image_size = (224, 224)  # Target image size

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,  # Random rotations
    brightness_range=[0.8, 1.2],  # Brightness adjustments
    horizontal_flip=True  # Horizontal flipping
)

# Function to preprocess and augment images
def preprocess_images():
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)

    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        save_label_path = os.path.join(processed_path, label)

        if not os.path.exists(save_label_path):
            os.makedirs(save_label_path)

        for file in os.listdir(label_path):
            img_path = os.path.join(label_path, file)
            img = cv2.imread(img_path)
            if img is None:
                continue  # Skip if the image cannot be loaded

            # Resize and normalize the image
            img_resized = cv2.resize(img, image_size)
            img_normalized = img_resized / 255.0

            # Save the original processed image
            save_path = os.path.join(save_label_path, file)
            cv2.imwrite(save_path, (img_normalized * 255).astype('uint8'))

            # Apply augmentation and save augmented images
            img_expanded = img_resized.reshape((1,) + img_resized.shape)
            for batch in datagen.flow(img_expanded, batch_size=1, save_to_dir=save_label_path, save_format='jpg'):
                break  # Generate one augmented image per original

# Run the preprocessing function
preprocess_images()
print("Preprocessing completed. Processed images are saved in:", processed_path)
