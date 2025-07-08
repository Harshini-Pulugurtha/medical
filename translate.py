import streamlit as st
import traceback
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

DB_FAISS_PATH = 'vectorstore/db_faiss'

# @st.cache_resource
# def get_translator(target_language="fr"):
#     model_map = {
#         "fr": "Helsinki-NLP/opus-mt-en-fr",
#         "es": "Helsinki-NLP/opus-mt-en-es",
#         "de": "Helsinki-NLP/opus-mt-en-de",
#         "hi": "Helsinki-NLP/opus-mt-en-hi",
#         "ta": "Helsinki-NLP/opus-mt-en-ta",
#         "zh": "Helsinki-NLP/opus-mt-en-zh",
#         "ja": "Helsinki-NLP/opus-mt-en-ja",
#         "ko": "Helsinki-NLP/opus-mt-en-ko",
#         "en": None  # No translation needed
#     }

#     model_name = model_map.get(target_language)
#     if model_name is None:
#         return None
#     return pipeline("translation", model=model_name)


# def translate_answer(answer, lang_code):
#     translator = get_translator(lang_code)
#     translated = translator(answer, max_length=512)
#     return translated[0]['translation_text']

# Custom prompt
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=['context', 'question']
    )

def retrieval_qa_chain(llm, prompt, db):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

def load_llm():
    return CTransformers(
        model="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        model_type="mistral",
        config={
            "temperature": 0.7,
            "max_new_tokens": 256
        }
    )

@st.cache_resource
def get_translator(target_language="fr"):
    model_map = {
        "fr": "Helsinki-NLP/opus-mt-en-fr",
        "es": "Helsinki-NLP/opus-mt-en-es",
        "de": "Helsinki-NLP/opus-mt-en-de",
        "hi": "Helsinki-NLP/opus-mt-en-hi",  # Hindi
        "zh": "Helsinki-NLP/opus-mt-en-zh"   # Chinese
    }
    model_name = model_map.get(target_language, "Helsinki-NLP/opus-mt-en-fr")
    return pipeline("translation", model=model_name)

def translate_answer(answer, lang_code):
    translator = get_translator(lang_code)
    translated = translator(answer, max_length=512)
    return translated[0]['translation_text']
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    prompt = set_custom_prompt()
    return retrieval_qa_chain(llm, prompt, db)

# --- Streamlit App ---
def main():
    st.title("🩺 Medical Chatbot")

    lang_code = st.selectbox(
    "Select language for the answer:",
    options=["en", "fr", "es", "de", "hi", "zh"],
    format_func=lambda x: {
        "en": "English",
        "fr": "French",
        "es": "Spanish",
        "de": "German",
        "hi": "Hindi",
        "zh": "Chinese"
    }[x]
)



    # Initialize session state for chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [("assistant", "Hi there! How can I help you today? 😊")]

    # Display chat history
    for role, msg in st.session_state.chat_messages:
        with st.chat_message(role):
            st.markdown(msg)

    # Chat input from user
    user_input = st.chat_input("Ask a medical question...")

    if user_input:
        # Show user input
        st.chat_message("user").markdown(user_input)
        st.session_state.chat_messages.append(("user", user_input))

        try:
            qa = qa_bot()
            response = qa.invoke({"query": user_input})
            original_answer = response.get("result", "🤔 No answer found.")
            if lang_code != "en":
                answer = translate_answer(original_answer, lang_code)
            else:
                answer = original_answer
        except Exception as e:
            answer = f"❌ Error: {e}"
            st.text(traceback.format_exc())

        # Show assistant response
        st.chat_message("assistant").markdown(answer)
        st.session_state.chat_messages.append(("assistant", answer))

if __name__ == "__main__":
    main()
