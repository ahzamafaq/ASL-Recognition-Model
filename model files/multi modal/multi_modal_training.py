import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report

# Paths
input_csv_path = r"C:\path\to\multi_modal_dataset.csv"
multi_modal_model_path = r"C:\path\to\multi_modal_model.h5"

# Load the multi-modal dataset
df = pd.read_csv(input_csv_path)

# Separate skeletal features, image paths, and labels
skeletal_features = df.iloc[:, :-2].values  # All columns except 'image_path' and 'label'
image_paths = df["image_path"].values  # Image paths
labels = df["label"].values  # Class labels

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Load and preprocess image data
def preprocess_images(image_paths):
    images = []
    for path in image_paths:
        image = cv2.imread(path)
        resized_image = cv2.resize(image, (224, 224))  # Resize to 224x224
        normalized_image = resized_image / 255.0  # Normalize pixel values
        images.append(normalized_image)
    return np.array(images)

image_features = preprocess_images(image_paths)

# Split the data into training and testing sets
X_skeletal_train, X_skeletal_test, X_image_train, X_image_test, y_train, y_test = train_test_split(
    skeletal_features, image_features, encoded_labels, test_size=0.2, random_state=42
)

# Standardize the skeletal features
scaler = StandardScaler()
X_skeletal_train = scaler.fit_transform(X_skeletal_train)
X_skeletal_test = scaler.transform(X_skeletal_test)

# Define the multi-modal model
def create_multi_modal_model(skeletal_input_dim, image_input_shape, num_classes):
    # Skeletal input branch
    skeletal_input = Input(shape=(skeletal_input_dim,), name="Skeletal_Input")
    x = Dense(128, activation="relu")(skeletal_input)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    skeletal_branch = Model(inputs=skeletal_input, outputs=x)

    # Image input branch
    image_input = Input(shape=image_input_shape, name="Image_Input")
    y = Conv2D(32, (3, 3), activation="relu")(image_input)
    y = MaxPooling2D((2, 2))(y)
    y = Conv2D(64, (3, 3), activation="relu")(y)
    y = MaxPooling2D((2, 2))(y)
    y = Flatten()(y)
    image_branch = Model(inputs=image_input, outputs=y)

    # Combine the two branches
    combined = concatenate([skeletal_branch.output, image_branch.output])
    z = Dense(128, activation="relu")(combined)
    z = Dropout(0.3)(z)
    z = Dense(num_classes, activation="softmax")(z)

    # Final model
    model = Model(inputs=[skeletal_branch.input, image_branch.input], outputs=z)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# Create the multi-modal model
skeletal_input_dim = X_skeletal_train.shape[1]
image_input_shape = X_image_train.shape[1:]  # (224, 224, 3)
num_classes = len(np.unique(y_train))
multi_modal_model = create_multi_modal_model(skeletal_input_dim, image_input_shape, num_classes)

# Train the model
history = multi_modal_model.fit(
    [X_skeletal_train, X_image_train], y_train,
    validation_data=([X_skeletal_test, X_image_test], y_test),
    epochs=30,
    batch_size=32,
    verbose=1
)

# Evaluate the model
test_loss, test_accuracy = multi_modal_model.evaluate([X_skeletal_test, X_image_test], y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the trained model
multi_modal_model.save(multi_modal_model_path)
print(f"Multi-modal model saved as {multi_modal_model_path}")

# Classification report
y_pred = np.argmax(multi_modal_model.predict([X_skeletal_test, X_image_test]), axis=1)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
