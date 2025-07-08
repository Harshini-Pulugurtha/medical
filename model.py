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
        model="models/mistral-7b-instruct-v0.2.Q2_K.gguf",  # Local path to your downloaded file
        model_type="mistral",
        config={
            "temperature": 0.7,
            "max_new_tokens": 256
        }
    )
    return llm

@st.cache_resource
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

def main():
    st.title("Medical Chatbot")
    st.write("Hi, Welcome to Medical Bot. What is your query?")

    user_query = st.text_input("Enter your question:")
    if st.button("Ask"):
        if user_query.strip() == "":
            st.warning("Please enter a question.")
        else:
            try:
                qa = qa_bot()
                # response = qa({"query": user_query})
                response = qa.invoke({"query": user_query})
                answer = response.get("result", "No answer found.")
                sources = response.get("source_documents", [])

                st.markdown("**Answer:**")
                st.write(answer)

                
            except Exception as e:
                st.error(f"⚠️ Something went wrong during processing: {e}")
                st.text(traceback.format_exc())

if __name__ == "__main__":
    main()