import cv2
import numpy as np
import datetime
from tensorflow.keras.models import load_model
import pyttsx3

# Load the trained model
model_path = r"C:\Users\akshi\Downloads\final_model.keras"
model = load_model(model_path)

# Define the class names
class_names = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", 
    "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "space", "nothing", "del"
]

# Initialize webcam (use index 0 for the external webcam as detected earlier)
cap = cv2.VideoCapture(0)  # Change to the correct camera index if needed
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Set webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize Text-to-Speech (TTS) engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Speed of speech
tts_engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)

# Open a log file to record predictions
log_file = open("asl_predictions_log.txt", "a")

# Preprocess the frame for model prediction
def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))  # Resize to match the model's input size
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Enhance low-light frames
def enhance_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert to HSV
    h, s, v = cv2.split(hsv)
    v = cv2.equalizeHist(v)  # Equalize the brightness channel
    hsv = cv2.merge((h, s, v))
    enhanced_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # Convert back to BGR
    return enhanced_frame

# Confidence threshold for predictions
confidence_threshold = 0.7

print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Enhance the frame for low-light conditions
    frame = enhance_frame(frame)

    # Define region of interest (ROI) for hand detection
    x1, y1, x2, y2 = 100, 100, 400, 400
    roi = frame[y1:y2, x1:x2]

    # Preprocess the ROI for prediction
    preprocessed_roi = preprocess_frame(roi)

    # Predict the class
    predictions = model.predict(preprocessed_roi)
    predicted_class_index = np.argmax(predictions)
    confidence = predictions[0][predicted_class_index]

    # Display prediction only if confidence is above the threshold
    if confidence > confidence_threshold:
        predicted_class = class_names[predicted_class_index]
        text = f"{predicted_class} ({confidence * 100:.2f}%)"
        
        # Speak the prediction
        tts_engine.say(predicted_class)
        tts_engine.runAndWait()

        # Log the prediction
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_file.write(f"{timestamp}: {predicted_class} ({confidence * 100:.2f}%)\n")
    else:
        text = "Uncertain"

    # Draw the ROI and display the prediction
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('ASL Real-Time Recognition', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
log_file.close()
