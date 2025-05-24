import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load the multi-modal model
model_path = r"C:\Users\akshi\Desktop\testing\final\large\non augmented\multi modal\multimodal_model_large_non_augmented.h5"
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

# Preprocess the ROI for the model
def preprocess_roi(roi, img_size=(224, 224)):
    roi = cv2.resize(roi, img_size)  # Resize to the model's input size
    roi = roi / 255.0  # Normalize pixel values to [0, 1]
    roi = np.expand_dims(roi, axis=0)  # Add batch dimension
    return roi

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

    # Define ROI coordinates
    roi_x1, roi_y1, roi_x2, roi_y2 = 100, 100, 400, 400
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extract skeletal landmarks (x, y coordinates)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
            
            # Convert skeletal landmarks to numpy array and reshape
            landmarks = np.array(landmarks).reshape(1, -1)

            # Preprocess the ROI
            preprocessed_roi = preprocess_roi(roi)

            # Make multi-modal prediction (inputs: [image_input, skeletal_input])
            predictions = model.predict([preprocessed_roi, landmarks])
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

    # Draw the ROI box
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
    cv2.putText(frame, "Place hand in the box", (roi_x1, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Multi-Modal ASL Recognition', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
