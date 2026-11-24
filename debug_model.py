
from pathlib import Path
import joblib

from features import extract_voice_features

BASE = Path(__file__).resolve().parent

model = joblib.load(
    BASE / "models" / "speaker_model.pkl"
)

folder = BASE / "recordings" / "processed" / "ravion"

print("\nTESTING RAVION'S ORIGINAL RECORDINGS\n")

for file in sorted(folder.glob("*.wav")):

    features = extract_voice_features(file)

    prediction = model.predict(
        features.reshape(1, -1)
    )[0]

    print(
        f"{file.name} -> Predicted: {prediction}"
    )
