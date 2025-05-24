import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load the skeletal-based model
model_path = r"C:\Users\akshi\Desktop\testing\final\large\non augmented\skeletal\skeletal_model_large_non_augmented.h5"
model = load_model(model_path)

# Define the class names (ensure these match your model's output labels)
class_names = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", 
    "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "space", "nothing", "del"
]

# Mediapipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Webcam initialization
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Flip the frame for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extract hand landmarks as a flattened list of x and y only
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
            
            # Convert landmarks to numpy array and reshape for the model
            landmarks = np.array(landmarks).reshape(1, -1)

            # Make prediction using the skeletal model
            predictions = model.predict(landmarks)
            predicted_class_index = np.argmax(predictions)
            predicted_class = class_names[predicted_class_index]
            confidence = np.max(predictions)

            # Display prediction and confidence
            text = f"{predicted_class} ({confidence * 100:.2f}%)"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        cv2.putText(frame, "No hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow('Skeletal-Based ASL Recognition', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
