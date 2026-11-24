
from pathlib import Path
import numpy as np
import joblib

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from features import extract_voice_features

# Locations
BASE = Path(__file__).resolve().parent
RECORDINGS = BASE / "recordings" / "processed"
MODEL_FOLDER = BASE / "models"

MODEL_FOLDER.mkdir(exist_ok=True)

speakers = ["ravion", "samantha", "karis", "fleezy"]

X_train = []
y_train = []
X_test = []
y_test = []

print("Loading voice recordings...")

for speaker in speakers:

    folder = RECORDINGS / speaker
    files = sorted(folder.glob("*.wav"))

    if len(files) < 5:
        raise ValueError(
            f"{speaker} needs at least 5 WAV recordings."
        )

    # First 4 recordings: training
    for file in files[:4]:
        features = extract_voice_features(file)

        X_train.append(features)
        y_train.append(speaker)

    # 5th recording: preliminary test
    test_features = extract_voice_features(files[4])

    X_test.append(test_features)
    y_test.append(speaker)

    print(f"{speaker}: 4 training, 1 test")

# Create our machine-learning model
model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=3)
)

# Teach the model using labeled voice features
model.fit(np.array(X_train), y_train)

print("\nTraining complete!")

# Test the model on unseen recordings
predictions = model.predict(np.array(X_test))

print("\nSPEAKER IDENTIFICATION RESULTS")

for actual, predicted in zip(y_test, predictions):
    print(f"Actual: {actual} | Predicted: {predicted}")

# Calculate preliminary accuracy
accuracy = accuracy_score(y_test, predictions)

print(f"\nPreliminary Accuracy: {accuracy * 100:.1f}%")

# Save the trained model
model_path = MODEL_FOLDER / "speaker_model.pkl"
joblib.dump(model, model_path)

print(f"\nModel saved to: {model_path}")
