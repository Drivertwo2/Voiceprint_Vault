
import streamlit as st
from pathlib import Path
import tempfile
import hashlib
import joblib

from features import extract_voice_features

# ==========================================
# VOICEPRINT VAULT
# Florida A&M University - EEL 4930
# Speaker Identification Using Speech Features
# ==========================================

# PAGE SETTINGS
st.set_page_config(
    page_title="Voiceprint Vault",
    page_icon="🎙️",
    layout="centered"
)

# PROJECT FILE LOCATIONS
BASE_FOLDER = Path(__file__).resolve().parent
MODEL_PATH = BASE_FOLDER / "models" / "speaker_model.pkl"


# ==========================================
# CUSTOM STYLING
# ==========================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(
        145deg,
        #201429,
        #44213e,
        #8b2d50
    );
    color: white;
}

/* Main content */
.block-container {
    max-width: 750px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Titles */
h1 {
    color: white !important;
    text-align: center;
    font-size: 48px !important;
}

h2, h3 {
    color: white !important;
}

/* Normal text */
p {
    color: #f6dce7;
}

/* Pink recording card */
.st-key-recorder_card {
    background: linear-gradient(
        145deg,
        #622547,
        #df416f
    );
    border-radius: 30px;
    padding: 35px;
    margin-bottom: 25px;
}

/* Recorder inside pink card */
.st-key-recorder_card [data-testid="stAudioInput"] {
    background: rgba(255,255,255,0.12);
    padding: 18px;
    border-radius: 20px;
}

/* Buttons */
.stButton button,
.stDownloadButton button {
    background: linear-gradient(
        90deg,
        #ff416c,
        #ff6b91
    );
    color: white;
    border: none;
    border-radius: 25px;
    padding: 12px 20px;
    font-weight: bold;
    width: 100%;
}

.stButton button:hover,
.stDownloadButton button:hover {
    background: #ff7b9b;
    color: white;
    border: none;
}

/* Speaker result card */
.st-key-result_card {
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.20);
    border-radius: 20px;
    padding: 20px;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# APP TITLE
# ==========================================

st.title("🎙️ Voiceprint Vault")

st.markdown(
    "<p style='text-align:center;font-size:18px;color:#f6dce7;'>"
    "Speaker Identification Using Speech Features"
    "</p>",
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# SESSION STATE
# ==========================================

# Allows the recording to be deleted and reset
if "recorder_version" not in st.session_state:
    st.session_state.recorder_version = 0

# Stores the most recent speaker prediction
if "predicted_speaker" not in st.session_state:
    st.session_state.predicted_speaker = None

# Keeps results linked to the correct recording
if "last_audio_signature" not in st.session_state:
    st.session_state.last_audio_signature = None


# ==========================================
# RECORDING CARD
# ==========================================

with st.container(key="recorder_card"):

    # Large decorative microphone
    st.markdown(
        '<div style="text-align:center;padding:20px 0;">'
        '<div style="width:170px;height:170px;'
        'background:#ff416c;border-radius:50%;'
        'margin:30px auto;display:flex;'
        'align-items:center;justify-content:center;'
        'font-size:85px;'
        'box-shadow:0 0 0 15px #ffffff15,'
        '0 0 0 30px #ffffff10;">'
        '🎙️'
        '</div>'
        '<h2 style="color:white;margin-top:45px;">'
        'Ready to Record'
        '</h2>'
        '<p style="color:#ffe4ed;">'
        'Your Voice. Your Identity.'
        '</p>'
        '</div>',
        unsafe_allow_html=True
    )

    # Working microphone recorder
    audio = st.audio_input(
        "🎤 Press the microphone below to start recording",
        sample_rate=16000,
        key=f"recording_{st.session_state.recorder_version}"
    )


# ==========================================
# CHECK THE CURRENT RECORDING
# ==========================================

if audio is not None:

    audio_bytes = audio.getvalue()

    # Detect when the user makes a different recording
    audio_signature = hashlib.sha256(
        audio_bytes
    ).hexdigest()

    if (
        st.session_state.last_audio_signature
        != audio_signature
    ):
        st.session_state.predicted_speaker = None
        st.session_state.last_audio_signature = audio_signature

else:
    audio_bytes = None
    st.session_state.predicted_speaker = None
    st.session_state.last_audio_signature = None


# ==========================================
# AUDIO PLAYBACK AND BUTTONS
# ==========================================

if audio is not None:

    st.success("✅ Recording Complete!")

    st.subheader("🔊 Your Recording")

    # Play recorded audio
    st.audio(audio_bytes, format="audio/wav")

    col1, col2 = st.columns(2)

    # DOWNLOAD RECORDING
    with col1:

        st.download_button(
            label="⬇️ Download Recording",
            data=audio_bytes,
            file_name="voice_recording.wav",
            mime="audio/wav",
            use_container_width=True
        )

    # DELETE RECORDING
    with col2:

        if st.button(
            "🗑️ Delete Recording",
            use_container_width=True
        ):

            st.session_state.recorder_version += 1
            st.session_state.predicted_speaker = None
            st.session_state.last_audio_signature = None

            st.rerun()


# ==========================================
# AI SPEAKER IDENTIFICATION
# ==========================================

st.divider()

st.subheader("🔐 Speaker Identification")

st.write(
    "Voiceprint Vault uses digital speech processing "
    "and machine learning to identify registered speakers."
)

# No recording available
if audio is None:

    st.info(
        "Record your voice above to begin speaker identification."
    )

# Recording is available
else:

    if st.button(
        "🔍 Identify Speaker",
        use_container_width=True,
        type="primary"
    ):

        # Make sure the trained model exists
        if not MODEL_PATH.exists():

            st.error(
                "Trained model not found. "
                "Run train.py before identifying a speaker."
            )

        else:

            temp_path = None
            st.session_state.predicted_speaker = None

            try:

                with st.spinner("Analyzing voice patterns..."):

                    # Save the current recording temporarily
                    with tempfile.NamedTemporaryFile(
                        suffix=".wav",
                        delete=False
                    ) as temp_file:

                        temp_file.write(audio_bytes)
                        temp_path = Path(temp_file.name)

                    # Extract the same MFCC features
                    # used when training the model
                    voice_features = extract_voice_features(
                        temp_path
                    )

                    # Load our trained speaker model
                    model = joblib.load(MODEL_PATH)

                    # Predict who is speaking
                    prediction = model.predict(
                        voice_features.reshape(1, -1)
                    )[0]

                    # Store the result
                    st.session_state.predicted_speaker = str(
                        prediction
                    ).title()

            except Exception as error:

                st.error(
                    f"Could not analyze the recording: {error}"
                )

            finally:

                # Delete temporary audio file
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)


# ==========================================
# DISPLAY SPEAKER RESULT
# ==========================================

if st.session_state.predicted_speaker is not None:

    with st.container(key="result_card"):

        st.markdown("### ✅ Speaker Identified")

        st.markdown("**Predicted Speaker:**")

        st.markdown(
            f"## 🎙️ {st.session_state.predicted_speaker}"
        )

        st.caption(
            "Prediction based on MFCC speech features "
            "and the trained k-nearest neighbors model."
        )

    st.success(
        "Voice analysis completed successfully!"
    )

    st.caption(
        "The system selects the closest match among "
        "the four registered speakers. It does not "
        "verify a person's identity for security purposes."
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Florida A&M University | EEL 4930 | "
    "Digital Speech Signal Processing"
)

st.caption(
    "Voiceprint Vault | Ravion Hyatt & Samantha"
)
