:import os 
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Adjust path based on your current working directory
DATASET_PATH = "useless-project-Zzzify/model/dataset"

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=16000)  # Ensure 16kHz sample rate
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

features = []
labels = []

# Update the path to reflect the correct directory structure
base_path = "useless-project-Zzzify/model/dataset"
for label, folder_name in enumerate(["snore", "non_snore"]):
    folder_path = os.path.join(base_path, folder_name)
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        continue
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".wav"):
            file_path = os.path.join(folder_path, file_name)
            try:
                data = extract_features(file_path)
                features.append(data)
                labels.append(label)
            except Exception as e:
                print(f"⚠️ Error processing {file_name}: {e}")

X = np.array(features)
# Corrected line: Explicitly set num_classes to 2 for one-hot encoding
y = to_categorical(np.array(labels), num_classes=2)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model
model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))

# Save model
model.save("snore_detector.h5")
print("✅ Model trained and saved as snore_detector.h5")