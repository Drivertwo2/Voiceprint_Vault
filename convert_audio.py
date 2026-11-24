
from pathlib import Path
import subprocess

# Find the recordings folder
project = Path(__file__).resolve().parent
recordings = project / "recordings"

# Folder for converted WAV files
output = recordings / "processed"

speakers = ["ravion", "samantha", "karis", "fleezy"]

for speaker in speakers:
    source_folder = recordings / speaker
    output_folder = output / speaker
    output_folder.mkdir(parents=True, exist_ok=True)

    if not source_folder.exists():
        print(f"Missing folder: {speaker}")
        continue

    # Find the original recordings
    files = sorted(
        [
            f for f in source_folder.iterdir()
            if f.suffix.lower() in [".m4a", ".wav"]
        ],
        key=lambda f: f.name.lower()
    )

    for number, file in enumerate(files, start=1):
        new_name = f"{speaker}_{number:02d}.wav"
        destination = output_folder / new_name

        if destination.exists():
            print(f"Already exists: {new_name}")
            continue

        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-n",
            "-i", str(file),
            "-map", "0:a:0",
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(destination)
        ]

        try:
            subprocess.run(command, check=True)
            print(f"Converted: {file.name} -> {new_name}")
        except subprocess.CalledProcessError:
            print(f"ERROR converting: {file.name}")

print("\nAudio conversion finished!")
