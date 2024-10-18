import os
import re
import streamlit as st
from groq import Groq
import numpy as np
from dotenv import load_dotenv
from bark import generate_audio, preload_models, SAMPLE_RATE
import soundfile as sf
import io

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["SUNO_USE_SMALL_MODELS"] = "1"

load_dotenv()
preload_models()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
st.set_page_config(page_title="Lecture Generator", page_icon=":book:")

def generate_lecture(topic):
    description = "You are an experienced educator tasked with creating a comprehensive lecture on the given topic."
    instructions = [
        "Create an engaging and informative lecture based on the provided topic.",
        "Structure the lecture with an introduction, main content divided into clear sections, and a conclusion.",
        "Include clear transitions between sections and use natural speech patterns.",
        "Incorporate pauses, emphasis, and tonal variations using Bark TTS markup.",
        "Use [PAUSE] for pauses, *text* for emphasis, and ALL CAPS for increased volume.",
        "Keep the total lecture length to about 5-7 minutes of spoken content.",
    ]
    lecture_format = """
    [INTRO]
    Hello, class! Today, we'll be discussing [TOPIC]. *This is an important subject* that [BRIEF REASON].
    
    [PAUSE]
    
    Let's begin by outlining our main points:
    1. [POINT 1]
    2. [POINT 2]
    3. [POINT 3]
    
    [MAIN CONTENT]
    Now, let's dive into our first point: [POINT 1].
    [EXPLAIN POINT 1]
    [PAUSE]
    
    Moving on to our second point: [POINT 2].
    [EXPLAIN POINT 2]
    [PAUSE]
    
    Finally, let's discuss our third point: [POINT 3].
    [EXPLAIN POINT 3]
    [PAUSE]
    
    [CONCLUSION]
    To summarize, we've covered [BRIEF RECAP].
    Remember, the KEY TAKEAWAY is [MAIN LESSON].
    
    Thank you for your attention, and I hope you found this lecture on [TOPIC] informative!
    """
    prompt = f"{description}\n\nInstructions:\n" + "\n".join(
        f"- {instruction}" for instruction in instructions
    )
    prompt += f"\n\nLecture Format:\n{lecture_format}\n\nTopic: {topic}\n\n"

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-8b-8192",
    )

    return chat_completion.choices[0].message.content

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
            lecture_content = generate_lecture(input_topic)
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