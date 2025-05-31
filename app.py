import streamlit as st
import google.generativeai as genai
import tempfile
import requests
from PIL import Image
import sounddevice as sd
import scipy.io.wavfile as wavfile


def transcribe_audio(file_path, sarvam_api_key):
    url = "https://api.sarvam.ai/speech-to-text"
    headers = {
        "api-subscription-key": sarvam_api_key,
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


st.set_page_config(page_title="Multimodal Chat", layout="wide")

st.markdown("<h1 style='text-align: center;'> AI BREWERY Multimodal Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Chat via <strong>Text</strong>, <strong>Voice</strong>, or <strong>Image</strong>!</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar inputs for API keys
with st.sidebar:
    st.markdown("## 🔑 Enter API Keys")
    gemini_api_key = st.text_input("Gemini API Key", type="password")
    sarvam_api_key = st.text_input("Sarvam STT API Key", type="password")
    st.markdown("---")
    st.markdown("## 🛠️ Input Options")
    uploaded_image = st.file_uploader("🖼️ Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        st.image(uploaded_image, caption="Preview", use_container_width=True)

if not gemini_api_key or not sarvam_api_key:
    st.warning("Please enter both Gemini and Sarvam API keys to start chatting.")
    st.stop()

# Configure Gemini API with user-provided key
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Voice recording and transcription
with st.sidebar:
    if st.button("🎤 Record Voice (5s)"):
        audio_path = record_voice()
        transcript = transcribe_audio(audio_path, sarvam_api_key)
        if transcript and not transcript.startswith("Error:"):
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

# Chat input and response
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

# Display conversation
st.markdown("---")
st.markdown(" Conversation")
chat_box = st.container()
with chat_box:
    for msg in st.session_state.messages:
        role = msg["role"]
        if role == "user":
            bubble_color = "#4169E1"
            text_color = "white"
        else:
            bubble_color = "#F1F0F0"
            text_color = "black"
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
