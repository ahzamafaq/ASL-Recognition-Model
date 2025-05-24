import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os

# Paths to dataset splits
train_path = "C:/Users/akshi/Desktop/STS/asl_alphabet_splits/train"
val_path = "C:/Users/akshi/Desktop/STS/asl_alphabet_splits/val"
test_path = "C:/Users/akshi/Desktop/STS/asl_alphabet_splits/test"

# Parameters
image_size = (224, 224)
batch_size = 32
epochs = 20

# Data Generators
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    val_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical"
)

# Debugging: Ensure correct number of samples
print(f"Total training samples: {train_generator.samples}")
print(f"Total validation samples: {val_generator.samples}")
print(f"Total test samples: {test_generator.samples}")

# Calculate steps per epoch dynamically
steps_per_epoch = train_generator.samples // batch_size
validation_steps = val_generator.samples // batch_size
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")

# Load MobileNetV2 model (pretrained on ImageNet)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers
base_model.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(train_generator.num_classes, activation="softmax")(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model (initial training)
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps
)

# Fine-tune the model
print("Unfreezing base model for fine-tuning...")
base_model.trainable = True

# Recompile the model with a lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Fine-tune training
fine_tune_history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps
)

# Evaluate on the test set
test_loss, test_acc = model.evaluate(test_generator)
print(f"Fine-Tuned Model Test Accuracy: {test_acc * 100:.2f}%")

# Save the model
model.save("asl_model_mobilenetv2.h5")
print("Model saved as 'asl_model_mobilenetv2.h5'")
