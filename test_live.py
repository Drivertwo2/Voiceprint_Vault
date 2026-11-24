
from pathlib import Path
import joblib
import numpy as np

from features import extract_voice_features

# Project locations
BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "speaker_model.pkl"
AUDIO_PATH = BASE / "live_tests" / "voice_recording.wav"

# Check that the recording exists
if not AUDIO_PATH.exists():
    raise FileNotFoundError(
        f"Could not find: {AUDIO_PATH}"
    )

# Load the trained model
model = joblib.load(MODEL_PATH)

# Extract features from live recording
features = extract_voice_features(AUDIO_PATH)

# Predict the speaker
prediction = model.predict(
    features.reshape(1, -1)
)[0]

print("\nLIVE RECORDING TEST")
print("-------------------------")
print("Predicted speaker:", prediction)

# Show how close the recording is to training samples
distances, indices = model.named_steps[
    "kneighborsclassifier"
].kneighbors(
    model.named_steps["standardscaler"].transform(
        features.reshape(1, -1)
    ),
    n_neighbors=5
)

training_labels = model.named_steps[
    "kneighborsclassifier"
]._y

speaker_names = model.classes_

print("\nFive closest training recordings:")

for distance, index in zip(
    distances[0], indices[0]
):
    speaker_number = training_labels[index]
    speaker_name = speaker_names[speaker_number]

    print(
        f"Speaker: {speaker_name} | "
        f"Distance: {distance:.2f}"
    )

print("\nTest complete!")
