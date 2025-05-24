import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Paths
input_csv_path = r"C:\path\to\skeletal_dataset.csv"
model_save_path = r"C:\path\to\skeletal_model.h5"
label_encoder_save_path = r"C:\path\to\label_encoder.pkl"
scaler_save_path = r"C:\path\to\scaler.pkl"

# Load the skeletal dataset
skeletal_df = pd.read_csv(input_csv_path)

# Separate features and labels
features = skeletal_df.drop(columns=["label"]).values  # All columns except 'label'
labels = skeletal_df["label"].values  # The 'label' column

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)  # Convert string labels to integers

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.2, random_state=42)

# Standardize the skeletal features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for future use
with open(scaler_save_path, "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

# Define the skeletal model (MLP)
def create_skeletal_model(input_dim, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')  # Output layer with softmax activation
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Create the model
input_dim = X_train.shape[1]
num_classes = len(np.unique(y_train))
skeletal_model = create_skeletal_model(input_dim, num_classes)

# Train the model
history = skeletal_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32,
    verbose=1
)

# Evaluate the model
test_loss, test_accuracy = skeletal_model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the trained model
skeletal_model.save(model_save_path)
print(f"Skeletal model saved as {model_save_path}")

# Save the label encoder
with open(label_encoder_save_path, "wb") as encoder_file:
    pickle.dump(label_encoder, encoder_file)

# Classification report
y_pred = np.argmax(skeletal_model.predict(X_test), axis=1)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
