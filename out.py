import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import note_seq
import tempfile
import subprocess
import os

# ---------------- CONFIG ----------------
MODEL_NAME = "nissan2323/finalmodel"
SOUNDFONT_PATH = "your/soundfont/file/path"
NOTE_LENGTH_16TH_120BPM = 0.25 * 60 / 120
BAR_LENGTH_120BPM = 4.0 * 60 / 120

# ----------------- FUNCTIONS -----------------

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    return model, tokenizer

def generate_tokens(start_seq, model, tokenizer, temperature=1.0, max_length=2048):
    input_ids = tokenizer.encode(start_seq, return_tensors="pt")
    eos_token_id = tokenizer.encode("TRACK_END")[0]
    generated_ids = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        temperature=temperature,
        eos_token_id=eos_token_id,
    )
    return tokenizer.decode(generated_ids[0])

def empty_note_sequence(qpm=120.0, total_time=0.0):
    note_sequence = note_seq.protobuf.music_pb2.NoteSequence()
    note_sequence.tempos.add().qpm = qpm
    note_sequence.ticks_per_quarter = note_seq.constants.STANDARD_PPQ
    note_sequence.total_time = total_time
    return note_sequence

def token_sequence_to_note_sequence(token_sequence):
    if isinstance(token_sequence, str):
        token_sequence = token_sequence.split()

    note_sequence = empty_note_sequence()

    current_program = 1
    current_is_drum = False
    current_instrument = 0
    track_count = 0
    instruments = {}

    for token_index, token in enumerate(token_sequence):
        if token == "PIECE_START":
            pass
        elif token == "PIECE_END":
            break
        elif token == "TRACK_START":
            current_bar_index = 0
            track_count += 1
            instruments[track_count] = {
                "program": current_program,
                "is_drum": current_is_drum,
                "instrument": track_count,
                "current_notes": {}
            }
        elif token == "TRACK_END":
            pass
        elif token.startswith("INST"):
            instrument = token.split("=")[-1]
            if instrument != "DRUMS":
                instruments[track_count]["program"] = int(instrument)
                instruments[track_count]["is_drum"] = False
            else:
                instruments[track_count]["program"] = 0
                instruments[track_count]["is_drum"] = True
        elif token == "BAR_START":
            instruments[track_count]["current_time"] = current_bar_index * BAR_LENGTH_120BPM
        elif token == "BAR_END":
            current_bar_index += 1
        elif token.startswith("NOTE_ON"):
            pitch = int(token.split("=")[-1])
            current_time = instruments[track_count]["current_time"]
            note = note_sequence.notes.add()
            note.start_time = current_time
            note.end_time = current_time + 4 * NOTE_LENGTH_16TH_120BPM
            note.pitch = pitch
            note.instrument = instruments[track_count]["instrument"]
            note.program = instruments[track_count]["program"]
            note.velocity = 80
            note.is_drum = instruments[track_count]["is_drum"]
            instruments[track_count]["current_notes"][pitch] = note
        elif token.startswith("NOTE_OFF"):
            pitch = int(token.split("=")[-1])
            if pitch in instruments[track_count]["current_notes"]:
                note = instruments[track_count]["current_notes"][pitch]
                note.end_time = instruments[track_count]["current_time"]
        elif token.startswith("TIME_DELTA"):
            delta = float(token.split("=")[-1]) * NOTE_LENGTH_16TH_120BPM
            instruments[track_count]["current_time"] += delta

    return note_sequence

def midi_to_wav(midi_path, wav_path, sf2_path):
    cmd = [
        "fluidsynth",
        "-ni",
        sf2_path,
        midi_path,
        "-F",
        wav_path,
        "-r",
        "44100"
    ]
    subprocess.run(cmd, check=True)

# ----------------- STREAMLIT UI -----------------

st.title("🎶 The Musical Virtuoso")
st.write("Generate music using your custom-trained model!")

model, tokenizer = load_model_and_tokenizer()

genre = st.selectbox("🎼 Select Genre", ["POP", "ROCK", "JAZZ"])
start_sequence = f"PIECE_START GENRE={genre} TRACK_START INST=0 DENSITY=3 BAR_START"

if "token_sequence" not in st.session_state:
    st.session_state.token_sequence = ""

if st.button("🎵 Generate New Music"):
    st.session_state.token_sequence = generate_tokens(start_sequence, model, tokenizer)

if st.button("🔁 Continue Generation"):
    st.session_state.token_sequence = generate_tokens(st.session_state.token_sequence, model, tokenizer)

if st.session_state.token_sequence:
    st.subheader("🧠 Generated Token Sequence")
    st.text_area("Tokens", st.session_state.token_sequence, height=200)

    # Token count
    token_count = len(st.session_state.token_sequence.split())
    st.markdown(f"**🎯 Total Tokens:** `{token_count}`")

    # Convert to NoteSequence
    note_sequence = token_sequence_to_note_sequence(st.session_state.token_sequence)

    # Save as MIDI
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp_midi:
        note_seq.sequence_proto_to_midi_file(note_sequence, tmp_midi.name)
        midi_path = tmp_midi.name

    # Convert to WAV
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        wav_path = tmp_wav.name
        midi_to_wav(midi_path, wav_path, SOUNDFONT_PATH)

    # Audio playback
    st.audio(wav_path)

    # Download button
    with open(wav_path, "rb") as f:
        st.download_button("⬇️ Download WAV", f, file_name="generated_music.wav", mime="audio/wav")
