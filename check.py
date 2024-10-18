import streamlit as st
import numpy as np
import re
from bark import generate_audio, preload_models, SAMPLE_RATE
import soundfile as sf
import io
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["SUNO_USE_SMALL_MODELS"] = "1"

preload_models()

def split_into_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text)

def generate_speech(lecture_content):
    sentences = split_into_sentences(lecture_content)
    audio_pieces = []
    
    for sentence in sentences:
        audio_array = generate_audio(sentence, history_prompt="v2/en_speaker_6")
        audio_pieces.append(audio_array)
        
        # Add a short pause between sentences
        pause = np.zeros(int(0.3 * SAMPLE_RATE))
        audio_pieces.append(pause)
    
    return np.concatenate(audio_pieces)

def main():
    st.title("Lecture Generator with Audio")

    input_topic = st.text_input("Enter a lecture topic")
    generate_lecture_btn = st.button("Generate Lecture")

    if generate_lecture_btn and input_topic:
        with st.spinner("Generating Lecture"):
            lecture_content =  "I have a silky smooth voice, and today I will tell you about the exercise regimen of the common sloth."
            st.markdown(lecture_content)

        with st.spinner("Generating Audio"):
            audio_array = generate_speech(lecture_content)
            
            # Convert the audio array to a wav file in memory
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, audio_array, SAMPLE_RATE, format='wav')
            audio_buffer.seek(0)

            # Display the audio using st.audio
            st.audio(audio_buffer, format='audio/wav')

    elif generate_lecture_btn:
        st.warning("Please enter a lecture topic.")

if __name__ == "__main__":
    main()