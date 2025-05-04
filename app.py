import streamlit as st
import google.generativeai as genai
import os
import tempfile
import requests
import numpy as np
from PIL import Image
from pydub import AudioSegment
from dotenv import load_dotenv
import sounddevice as sd
import scipy.io.wavfile as wavfile


load_dotenv()
genai.configure(api_key=os.getenv("API_KEY"))
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")


model = genai.GenerativeModel("gemini-1.5-flash")


def transcribe_audio(file_path):
    url = "https://api.sarvam.ai/speech-to-text"
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Accept": "application/json"
    }
    data = {
        "language_code": "en-IN",
        "model": "saarika:v2",
        "with_timestamps": "false"
    }
    files = {
        "file": (file_path, open(file_path, "rb"), "audio/wav")
    }
    response = requests.post(url, headers=headers, files=files, data=data)
    if response.status_code == 200:
        return response.json().get("transcript", "")
    else:
        return f"Error: {response.text}"


def record_voice(duration=5, sample_rate=16000):
    st.toast("🎙️ Recording...", icon="🎙️")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wavfile.write(temp_file.name, sample_rate, audio)
    st.toast("✅ Done recording", icon="✅")
    return temp_file.name


st.set_page_config(page_title=" Multimodal Chat", layout="wide")
st.markdown("<h1 style='text-align: center;'> AI BREWERY Multimodal Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Chat via <strong>Text</strong>, <strong>Voice</strong>, or <strong>Image</strong>!</p>", unsafe_allow_html=True)
st.markdown("---")


if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.markdown("## 🛠️ Input Options")
    uploaded_image = st.file_uploader("🖼️ Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        st.image(uploaded_image, caption="Preview", use_container_width=True)

    if st.button("🎤 Record Voice (5s)"):
        audio_path = record_voice()
        transcript = transcribe_audio(audio_path)
        if transcript:
            st.session_state.messages.append({"role": "user", "content": transcript})
            contents = [transcript]
            if uploaded_image:
                contents.append(Image.open(uploaded_image))

            with st.spinner("Gemini is replying..."):
                response = model.generate_content(contents)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
            st.rerun()
        else:
            st.warning("❌ Failed to transcribe audio.")


st.subheader("💬 Chat with Gemini")
with st.form(key="chat_form", clear_on_submit=True):
    user_text = st.text_input("", placeholder="Type your message here...", label_visibility="collapsed")
    submitted = st.form_submit_button("Send")
    if submitted and user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})
        contents = [user_text]
        if uploaded_image:
            contents.append(Image.open(uploaded_image))

        with st.spinner("Gemini is replying..."):
            response = model.generate_content(contents)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
        st.rerun()


st.markdown("---")
st.markdown(" Conversation")
chat_box = st.container()
with chat_box:
    for msg in st.session_state.messages:
        role = msg["role"]
        # New colors: royal blue with white text for user, light gray with black text for assistant
        if role == "user":
            bubble_color = "#4169E1"  # Royal blue for user
            text_color = "white"       # White text
        else:
            bubble_color = "#F1F0F0"  # Light gray for assistant
            text_color = "black"       # Black text
        sender = "You" if role == "user" else "Chatbot"
        st.markdown(
            f"""
            <div style="background-color: {bubble_color}; padding: 10px 15px; border-radius: 10px; margin-bottom: 10px; max-width: 75%; color: {text_color};">
                <strong>{sender}:</strong><br>{msg['content']}
            </div>
            """,
            unsafe_allow_html=True
        )


st.markdown("---")
st.markdown("<center>💡 Powered by Gemini API + Sarvam STT</center>", unsafe_allow_html=True)
