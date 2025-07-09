import streamlit as st
import traceback
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import base64
import tempfile
from pydub import AudioSegment
import time
from dotenv import load_dotenv
import os

# ---------------- Configuration ---------------- #
load_dotenv()
GOOGLE_API_KEY = os.getenv("API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
DB_FAISS_PATH = 'vectorstore/db_faiss'

# ---------------- Language Maps ---------------- #
language_map = {
    "en": "English", "fr": "French", "es": "Spanish",
    "de": "German", "hi": "Hindi", "zh": "Chinese"
}
asr_lang_map = {
    "en": "en-US", "fr": "fr-FR", "es": "es-ES",
    "de": "de-DE", "hi": "hi-IN", "zh": "zh-CN"
}

# ---------------- Translation Pipelines ---------------- #
@st.cache_resource
def get_output_translator(target_language="fr"):
    model_map = {
        "fr": "Helsinki-NLP/opus-mt-en-fr",
        "es": "Helsinki-NLP/opus-mt-en-es",
        "de": "Helsinki-NLP/opus-mt-en-de",
        "hi": "Helsinki-NLP/opus-mt-en-hi",
        "zh": "Helsinki-NLP/opus-mt-en-zh"
    }
    model_name = model_map.get(target_language)
    return pipeline("translation", model=model_name) if model_name else None

@st.cache_resource
def get_input_translator(source_language="hi"):
    model_map = {
        "fr": "Helsinki-NLP/opus-mt-fr-en",
        "es": "Helsinki-NLP/opus-mt-es-en",
        "de": "Helsinki-NLP/opus-mt-de-en",
        "hi": "Helsinki-NLP/opus-mt-hi-en",
        "zh": "Helsinki-NLP/opus-mt-zh-en"
    }
    model_name = model_map.get(source_language)
    return pipeline("translation", model=model_name) if model_name else None

def translate_question(text, lang_code):
    if lang_code == "en":
        return text
    translator = get_input_translator(lang_code)
    translated = translator(text, max_length=512)
    return translated[0]['translation_text']

def translate_answer(text, lang_code):
    if lang_code == "en":
        return text
    translator = get_output_translator(lang_code)
    translated = translator(text, max_length=512)
    return translated[0]['translation_text']

# ---------------- Gemini Integration ---------------- #
@st.cache_resource
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

def ask_gemini(question, context):
    prompt = f"""You are a helpful medical assistant. Use the following context to answer the user's question.
If you do not know the answer, say "I don't know" instead of making up an answer.

Context: {context}

Question: {question}

Answer:"""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.7,
            "max_output_tokens": 512
        }
    )
    return response.text.strip()

# ---------------- Audio Utilities ---------------- #
def transcribe_audio(audio_bytes, language_code="en-US"):
    recognizer = sr.Recognizer()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_file:
            audio = AudioSegment.from_file(BytesIO(audio_bytes))
            audio.export(wav_file.name, format="wav")
            wav_path = wav_file.name

        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language=language_code)
            return text
    except Exception as e:
        return f"❌ Audio processing error: {e}"

def text_to_speech(text, lang_code):
    try:
        tts = gTTS(text=text, lang=lang_code)
        with BytesIO() as audio_file:
            tts.write_to_fp(audio_file)
            audio_file.seek(0)
            audio_bytes = audio_file.read()
        b64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
            <audio autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
        time.sleep(1)
    except Exception as e:
        st.error(f"Audio playback failed: {e}")

# ---------------- Main App ---------------- #
def main():
    st.title("🩺Medical Chatbot")

    output_lang = st.selectbox("🌐 Select language for response:", list(language_map.keys()), format_func=lambda x: language_map[x])
    input_lang = st.selectbox("🎙️ Select voice input language:", list(language_map.keys()), format_func=lambda x: language_map[x])

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [("assistant", "Hi there! How can I help you today? 😊")]

    for role, msg in st.session_state.chat_messages:
        with st.chat_message(role):
            st.markdown(msg)

    db = load_vector_store()

    user_input = st.chat_input("Type your medical question...")
    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.chat_messages.append(("user", user_input))

        try:
            translated_question = translate_question(user_input, output_lang)
            docs = db.similarity_search(translated_question, k=2)
            context = "\n\n".join(doc.page_content for doc in docs)
            original_answer = ask_gemini(translated_question, context)
            answer = translate_answer(original_answer, output_lang)
        except Exception as e:
            answer = f"❌ Error: {e}"
            st.text(traceback.format_exc())

        st.chat_message("assistant").markdown(answer)
        st.session_state.chat_messages.append(("assistant", answer))
        text_to_speech(answer, output_lang)

    audio = mic_recorder(start_prompt="🎤 Click to speak", stop_prompt="⏹️ Stop", just_once=True, key="voice")
    if audio:
        with st.spinner("Transcribing..."):
            spoken_text = transcribe_audio(audio["bytes"], asr_lang_map[input_lang])
            st.chat_message("user").markdown(spoken_text)
            st.session_state.chat_messages.append(("user", spoken_text))

            try:
                translated_question = translate_question(spoken_text, input_lang)
                docs = db.similarity_search(translated_question, k=2)
                context = "\n\n".join(doc.page_content for doc in docs)
                original_answer = ask_gemini(translated_question, context)
                answer = translate_answer(original_answer, output_lang)
            except Exception as e:
                answer = f"❌ Error: {e}"
                st.text(traceback.format_exc())

            st.chat_message("assistant").markdown(answer)
            st.session_state.chat_messages.append(("assistant", answer))
            text_to_speech(answer, output_lang)

if __name__ == "__main__":
    main()
