from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import streamlit as st
import traceback
from langchain_huggingface import HuggingFaceEmbeddings

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=['context', 'question']
    )
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def load_llm():
    llm = CTransformers(
        model="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        model_type="mistral",
        config={
            "temperature": 0.7,
            "max_new_tokens": 256}
        )
    return llm

@st.cache_resource
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    prompt = set_custom_prompt()
    return retrieval_qa_chain(llm, prompt, db)

# --- Streamlit App ---
def main():
    st.title("🩺 Medical Chatbot")

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
            answer = response.get("result", "🤔 No answer found.")
        except Exception as e:
            answer = f"❌ Error: {e}"
            st.text(traceback.format_exc())

        # Show assistant response
        st.chat_message("assistant").markdown(answer)
        st.session_state.chat_messages.append(("assistant", answer))

if __name__ == "__main__":
    main()
