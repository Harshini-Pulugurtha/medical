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
import os
import time
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

DB_FAISS_PATH = 'vectorstore/db_faiss'

language_map = {
    "en": "English", "fr": "French", "es": "Spanish",
    "de": "German", "hi": "Hindi", "zh": "Chinese",
    "ta": "Tamil", "bn": "Bengali", "ur": "Urdu"
}
asr_lang_map = {
    "en": "en-US", "fr": "fr-FR", "es": "es-ES",
    "de": "de-DE", "hi": "hi-IN", "zh": "zh-CN",
    "ta": "ta-IN", "bn": "bn-IN", "ur": "ur-IN"
}

sample_prompt = """You are a medical practitioner and an expert in analyzing medical-related images..."""

@st.cache_resource
def get_output_translator(target_language="fr"):
    model_map = {
        "fr": "Helsinki-NLP/opus-mt-en-fr",
        "es": "Helsinki-NLP/opus-mt-en-es",
        "de": "Helsinki-NLP/opus-mt-en-de",
        "hi": "Helsinki-NLP/opus-mt-en-hi",
        "zh": "Helsinki-NLP/opus-mt-en-zh",
        "ta": "Helsinki-NLP/opus-mt-en-ta",
        "bn": "Helsinki-NLP/opus-mt-en-bn",
        "ur": "Helsinki-NLP/opus-mt-en-ur"
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
        "zh": "Helsinki-NLP/opus-mt-zh-en",
        "ta": "Helsinki-NLP/opus-mt-ta-en",
        "bn": "Helsinki-NLP/opus-mt-bn-en",
        "ur": "Helsinki-NLP/opus-mt-ur-en"
    }
    model_name = model_map.get(source_language)
    return pipeline("translation", model=model_name) if model_name else None

def translate_question(text, lang_code):
    if lang_code == "en": return text
    translator = get_input_translator(lang_code)
    return translator(text, max_length=512)[0]['translation_text']

def translate_answer(text, lang_code):
    if lang_code == "en": return text
    translator = get_output_translator(lang_code)
    return translator(text, max_length=512)[0]['translation_text']

@st.cache_resource
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

def call_gemini_model_for_analysis(image_path: str, prompt: str = sample_prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    image = Image.open(image_path)
    response = model.generate_content([prompt, image], generation_config={"max_output_tokens": 1500})
    return response.text

def chat_eli(query: str):
    model = genai.GenerativeModel("gemini-1.5-flash")
    eli5_prompt = "You have to explain the below piece of information to a five years old:\n" + query
    response = model.generate_content(eli5_prompt)
    return response.text

def ask_gemini(question, context, simple=False, tone="formal"):
    tone_instruction = {
        "formal": "Provide a professional and detailed medical answer.",
        "friendly": "Answer in a clear, warm, and friendly tone.",
        "child": "Explain in very simple words like you're talking to a 10-year-old."
    }
    simplification = "\nExplain it in simple terms at the end." if simple else ""
    prompt = f"""You are a helpful multilingual medical assistant. Use the context below to answer.\n\nContext: {context}\n\nQuestion: {question}\n\n{tone_instruction.get(tone, '')}{simplification}\n\nAnswer:"""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt, generation_config={"temperature": 0.7, "max_output_tokens": 512})
    return response.text.strip()

def transcribe_audio(audio_bytes, language_code="en-US"):
    recognizer = sr.Recognizer()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_file:
            audio = AudioSegment.from_file(BytesIO(audio_bytes))
            audio.export(wav_file.name, format="wav")
        with sr.AudioFile(wav_file.name) as source:
            audio_data = recognizer.record(source)
            return recognizer.recognize_google(audio_data, language=language_code)
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
        st.markdown(f"""
            <audio autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        """, unsafe_allow_html=True)
        time.sleep(1)
    except Exception as e:
        st.error(f"Audio playback failed: {e}")

def main():
    st.title("🧠 Medical Image Analyzer + Chatbot")
    output_lang = st.selectbox("🌐 Response language:", list(language_map.keys()), format_func=lambda x: language_map[x])
    input_lang = st.selectbox("🎙️ Voice input language:", list(language_map.keys()), format_func=lambda x: language_map[x])
    simple_explanation = st.checkbox("📖 Simple mode (Explain like I'm 5)")
    tone = st.selectbox("🧘 Tone:", ["formal", "friendly", "child"])

    db = load_vector_store()

    # Upload image
    uploaded_file = st.file_uploader("Upload a medical image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name
        st.image(uploaded_file, caption="Uploaded Image")

        if st.button("🔬 Analyze Image"):
            try:
                result = call_gemini_model_for_analysis(file_path)
                st.session_state.image_analysis = result
                st.markdown(result)
            except Exception as e:
                st.error("Image analysis failed.")

    if st.session_state.get("image_analysis"):
        st.info("Would you like a simpler explanation?")
        if st.radio("ELI5?", ("No", "Yes")) == "Yes":
            st.markdown(chat_eli(st.session_state.image_analysis))

    # Start chatbot after image analysis
    if st.session_state.get("image_analysis"):
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = [("assistant", "Do you have any questions based on the analysis above or something else?")]

        for role, msg in st.session_state.chat_messages:
            with st.chat_message(role):
                st.markdown(msg)

        user_input = st.chat_input("Ask a follow-up or new medical question...")
        if user_input:
            st.chat_message("user").markdown(user_input)
            st.session_state.chat_messages.append(("user", user_input))
            try:
                translated_question = translate_question(user_input, output_lang)
                docs = db.similarity_search(translated_question, k=2)
                context = st.session_state.image_analysis + "\n\n" + "\n\n".join(doc.page_content for doc in docs)
                original_answer = ask_gemini(translated_question, context, simple=simple_explanation, tone=tone)
                answer = translate_answer(original_answer, output_lang)
            except Exception as e:
                answer = f"❌ Error: {e}"
                st.text(traceback.format_exc())
            st.chat_message("assistant").markdown(answer)
            st.session_state.chat_messages.append(("assistant", answer))
            text_to_speech(answer, output_lang)

        audio = mic_recorder(start_prompt="🎤 Speak", stop_prompt="⏹️ Stop", just_once=True, key="voice")
        if audio:
            spoken_text = transcribe_audio(audio["bytes"], asr_lang_map[input_lang])
            st.chat_message("user").markdown(spoken_text)
            st.session_state.chat_messages.append(("user", spoken_text))
            try:
                translated_question = translate_question(spoken_text, input_lang)
                docs = db.similarity_search(translated_question, k=2)
                context = st.session_state.image_analysis + "\n\n" + "\n\n".join(doc.page_content for doc in docs)
                original_answer = ask_gemini(translated_question, context, simple=simple_explanation, tone=tone)
                answer = translate_answer(original_answer, output_lang)
            except Exception as e:
                answer = f"❌ Error: {e}"
                st.text(traceback.format_exc())
            st.chat_message("assistant").markdown(answer)
            st.session_state.chat_messages.append(("assistant", answer))
            text_to_speech(answer, output_lang)

if __name__ == "__main__":
    main()
