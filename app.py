import os
import streamlit as st
from groq import Groq
import nltk
import numpy as np
from dotenv import load_dotenv
from bark.generation import (
    generate_text_semantic,
)
from bark.api import semantic_to_waveform
from bark import generate_audio, SAMPLE_RATE
from IPython.display import Audio

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
st.set_page_config(page_title="Course Generator", page_icon=":book:")


def generate_course(topic, search_results):
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
    prompt += (
        f"\n\nCourse Format:\n{course_format}\n\nTopic: {topic}\n\nSearch Results:\n"
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Explain the importance of fast language models",
            }
        ],
        model="llama3-8b-8192",
    )

    return chat_completion.text


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
            min_eos_p=0.05,  # this controls how likely the generation is to end
        )
    
    audio_array = semantic_to_waveform(semantic_tokens, history_prompt=SPEAKER,)
    pieces += [audio_array, silence.copy()]
    Audio(np.concatenate(pieces), rate=SAMPLE_RATE)

def main():

    input_topic = st.text_input("Enter a course topic")
    generate_course_btn = st.button("Generate Course")

    if generate_course_btn and input_topic:
        with st.spinner("Generating Course"):
            final_course = generate_course(input_topic)


    elif generate_course_btn:
        st.warning("Please enter a course topic.")


if __name__ == "__main__":
    main()
