import numpy as np
import librosa
from tensorflow.keras.models import load_model
import os

# Function to extract features from an audio file, just like during training
def extract_features(file_path):
    """
    Extracts MFCC features from an audio file.
    """
    try:
        audio, sr = librosa.load(file_path, sr=16000)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"⚠️ Error loading audio file {file_path}: {e}")
        return None

# Load the trained model
try:
    model = load_model("snore_detector.h5")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Set the path to a specific audio file to test
TEST_AUDIO_PATH = "c:\\users\\dell\\onedrive\\desktop\\Zzzify-web\\useless-project-Zzzify\\model\\dataset\\non snore\\alien-talking-312011.wav"

if os.path.exists(TEST_AUDIO_PATH):
    # Extract features from the test audio file
    features = extract_features(TEST_AUDIO_PATH)
    
    if features is not None:
        # Reshape the features to match the model's input shape (1, 40)
        features = features.reshape(1, -1)
        
        # Make a prediction
        prediction = model.predict(features)
        
        # Get the predicted class (0 for snore, 1 for non_snore)
        predicted_class = np.argmax(prediction)
        
        # Get the confidence for each class
        snore_confidence = prediction[0][0]
        non_snore_confidence = prediction[0][1]
        
        class_labels = ["Snore 😴", "Non-Snore ✅"]
        
        print("\n--- Prediction Results ---")
        print(f"File: {os.path.basename(TEST_AUDIO_PATH)}")
        print(f"Raw Prediction Array: {prediction}")
        print(f"Confidence for Snore: {snore_confidence:.2f}")
        print(f"Confidence for Non-Snore: {non_snore_confidence:.2f}")
        print(f"Predicted Label: {class_labels[predicted_class]}")
        print("--------------------------")
    
else:
    print(f"❌ Test file not found at: {TEST_AUDIO_PATH}")
    print("Please update the TEST_AUDIO_PATH variable to a valid audio file.")