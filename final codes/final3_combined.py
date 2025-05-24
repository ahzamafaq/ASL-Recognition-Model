import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import pyttsx3

# Load all models
skeletal_model_path = r"C:\Users\akshi\Desktop\testing\final\large\non augmented\skeletal\skeletal_model_large_non_augmented.h5"
multi_modal_model_path = r"C:\Users\akshi\Desktop\testing\final\large\non augmented\multi modal\multimodal_model_large_non_augmented.h5"
baseline_model_path = r"C:\Users\akshi\Downloads\final_model.keras"

skeletal_model = load_model(skeletal_model_path)
multi_modal_model = load_model(multi_modal_model_path)
baseline_model = load_model(baseline_model_path)

# Define class names (same for all models)
class_names = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", 
    "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "space", "nothing", "del"
]

# Mediapipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Text-to-Speech (TTS) setup
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Speed of speech
tts_engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)

# Webcam initialization
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Preprocess the ROI for image-based models
def preprocess_roi(roi, img_size=(224, 224)):
    roi = cv2.resize(roi, img_size)  # Resize to the model's input size
    roi = roi / 255.0  # Normalize pixel values to [0, 1]
    roi = np.expand_dims(roi, axis=0)  # Add batch dimension
    return roi

# Enhance low-light frames
def enhance_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert to HSV
    h, s, v = cv2.split(hsv)
    v = cv2.equalizeHist(v)  # Equalize the brightness channel
    hsv = cv2.merge((h, s, v))
    enhanced_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # Convert back to BGR
    return enhanced_frame

# Confidence threshold for filtering
confidence_threshold = 0.7

print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Flip the frame for a mirror effect
    frame = cv2.flip(frame, 1)

    # Enhance the frame for low-light conditions
    frame = enhance_frame(frame)

    # Convert the frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks
    result = hands.process(rgb_frame)

    # Define ROI coordinates
    roi_x1, roi_y1, roi_x2, roi_y2 = 100, 100, 400, 400
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

    skeletal_prediction = None
    baseline_prediction = None
    multi_modal_prediction = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extract skeletal landmarks (x, y coordinates)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
            
            # Convert skeletal landmarks to numpy array and reshape
            landmarks = np.array(landmarks).reshape(1, -1)

            # Preprocess the ROI for image-based models
            preprocessed_roi = preprocess_roi(roi)

            # Skeletal model prediction
            skeletal_predictions = skeletal_model.predict(landmarks)
            skeletal_prediction = {
                "class": class_names[np.argmax(skeletal_predictions)],
                "confidence": np.max(skeletal_predictions)
            }

            # Multi-modal model prediction
            multi_modal_predictions = multi_modal_model.predict([preprocessed_roi, landmarks])
            multi_modal_prediction = {
                "class": class_names[np.argmax(multi_modal_predictions)],
                "confidence": np.max(multi_modal_predictions)
            }

            # Baseline image model prediction
            baseline_predictions = baseline_model.predict(preprocessed_roi)
            baseline_prediction = {
                "class": class_names[np.argmax(baseline_predictions)],
                "confidence": np.max(baseline_predictions)
            }

            # Choose the prediction with the highest confidence
            all_predictions = [skeletal_prediction, baseline_prediction, multi_modal_prediction]
            best_prediction = max(all_predictions, key=lambda x: x["confidence"])

            if best_prediction["confidence"] > confidence_threshold:
                text = f"{best_prediction['class']} ({best_prediction['confidence'] * 100:.2f}%)"
                
                # Text-to-Speech for the final prediction
                tts_engine.say(best_prediction['class'])
                tts_engine.runAndWait()
            else:
                text = "Uncertain"

            # Display the chosen prediction
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        cv2.putText(frame, "No hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Draw the ROI box
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
    cv2.putText(frame, "Place hand in the box", (roi_x1, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('ASL Recognition with All Models', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
