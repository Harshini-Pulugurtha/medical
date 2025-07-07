from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import streamlit as st
import traceback

DB_FAISS_PATH = "vectorstore/db_faiss"

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}
Only returns the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

def load_llm():
    try:
        # Attempt to load the model
        llm = CTransformers(
            model="models/llama-2-7b-chat.Q4_K_M.gguf",
            model_type="llama_cpp",  
            config={'max_new_tokens': 512, 'temperature': 0.7}
        )
        return llm
    except Exception as e:
        st.error("❌ Failed to load the language model. Check the model path and format.")
        traceback.print_exc()
        raise e  # Re-raise for visibility in Streamlit

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

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
                response = qa({"query": user_query})
                answer = response.get("result", "No answer found.")
                sources = response.get("source_documents", [])

                st.markdown("**Answer:**")
                st.write(answer)

                if sources:
                    st.markdown("**Sources:**")
                    for i, doc in enumerate(sources, 1):
                        st.write(f"{i}. {getattr(doc, 'metadata', doc)}")
                else:
                    st.write("No sources found.")
            except Exception as e:
                st.error("⚠️ Something went wrong during processing. Check logs.")
                traceback.print_exc()

if __name__ == "__main__":
    main()
