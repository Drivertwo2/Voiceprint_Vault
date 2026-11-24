
import librosa
import numpy as np
from pathlib import Path

# Location of converted recordings
BASE_FOLDER = Path(__file__).resolve().parent
RECORDINGS = BASE_FOLDER / "recordings" / "processed"


def extract_voice_features(file_path):

    # Step 1: Load audio at 16,000 samples per second
    audio, sr = librosa.load(file_path, sr=16000, mono=True)

    # Step 2: Remove long silence at beginning and end
    audio, _ = librosa.effects.trim(audio, top_db=30)

    # Check recording
    if len(audio) < sr:
        raise ValueError("Recording is too short.")

    # Step 3: Extract 13 MFCC coefficients
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13,
        n_fft=512,
        hop_length=160
    )

    # Step 4: Calculate averages and standard deviations
    averages = np.mean(mfcc, axis=1)
    variations = np.std(mfcc, axis=1)

    # Step 5: Combine into 26 numbers
    features = np.concatenate((averages, variations))

    return features


# Test the program on our four speakers
if __name__ == "__main__":

    speakers = ["ravion", "samantha", "karis", "fleezy"]

    for speaker in speakers:

        folder = RECORDINGS / speaker
        files = sorted(folder.glob("*.wav"))

        print(f"\nSpeaker: {speaker}")
        print(f"WAV files found: {len(files)}")

        if not files:
            print("No converted WAV files found.")
            continue

        try:
            # Analyze the first recording
            features = extract_voice_features(files[0])

            print("Recording:", files[0].name)
            print("Features:", np.round(features, 2))
            print("Total features:", len(features))
            print("Analysis successful!")

        except Exception as error:
            print("Error:", error)
