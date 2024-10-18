import os
import streamlit as st
from groq import Groq
import nltk
import numpy as np
from dotenv import load_dotenv
from bark.generation import generate_text_semantic
from bark.api import semantic_to_waveform
from bark import SAMPLE_RATE
import soundfile as sf
import io

nltk.download('punkt')

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
st.set_page_config(page_title="Course Generator", page_icon=":book:")

def generate_course(topic):
    description = "You are an experienced educator tasked with creating a comprehensive course outline on the given topic."
    instructions = [
        "Create an engaging and informative course lecture based on the provided topic.",
        "The course should be structured like a speechable lecture, with given format.",
        "Include an introduction, learning objectives, 4-5 main modules with 3-5 subtopics each, and a conclusion.",
        "For each concept or topic, provide a brief explanation followed by a relevant YouTube source link to learn.",
        "Ensure the content is substantial, informative, and well-organized, resembling a full course page.",
        "Only output it as text.",
    ]
    course_format = """
    
    """
    prompt = f"{description}\n\nInstructions:\n" + "\n".join(
        f"- {instruction}" for instruction in instructions
    )
    prompt += f"\n\nCourse Format:\n{course_format}\n\nTopic: {topic}\n\n"

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

def generate_speech(final_course):
    sentences = nltk.sent_tokenize(final_course)
    GEN_TEMP = 0.6
    SPEAKER = "v2/en_speaker_6"
    silence = np.zeros(int(0.25 * SAMPLE_RATE))
    pieces = []
    for sentence in sentences:
        semantic_tokens = generate_text_semantic(
            sentence,
            history_prompt=SPEAKER,
            temp=GEN_TEMP,
            min_eos_p=0.05,
        )
        audio_array = semantic_to_waveform(semantic_tokens, history_prompt=SPEAKER)
        pieces += [audio_array, silence.copy()]
    
    return np.concatenate(pieces)

def main():
    st.title("Course Generator with Audio")

    input_topic = st.text_input("Enter a course topic")
    generate_course_btn = st.button("Generate Course")

    if generate_course_btn and input_topic:
        with st.spinner("Generating Course"):
            final_course = generate_course(input_topic)
            st.write(final_course)

        with st.spinner("Generating Audio"):
            audio_array = generate_speech(final_course)
            
            # Convert the audio array to a wav file in memory
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, audio_array, SAMPLE_RATE, format='wav')
            audio_buffer.seek(0)

            # Display the audio using st.audio
            st.audio(audio_buffer, format='audio/wav')

    elif generate_course_btn:
        st.warning("Please enter a course topic.")

if __name__ == "__main__":
    main()