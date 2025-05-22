import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import time
import tempfile

# Load NVIDIA API Key
load_dotenv()
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

# App Title and Description
st.title("🤖 Ask Your Docs – Powered by NVIDIA NIM 📄")
st.markdown("Upload a PDF and ask questions based on its content 💬")

# File Upload
uploaded_files = st.file_uploader("📁 Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)

# Initialize LLM and Prompt
llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct")
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Process files if uploaded
if uploaded_files and "vectors" not in st.session_state:
    with st.spinner("🔄 Processing uploaded documents..."):
        all_docs = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                loader = PyPDFLoader(tmp.name)
                docs = loader.load()
                all_docs.extend(docs)

        embeddings = NVIDIAEmbeddings()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        final_documents = text_splitter.split_documents(all_docs)
        st.session_state.vectors = FAISS.from_documents(final_documents, embeddings)

    st.success("Document processing complete. You can now ask questions.")

# Input box for the user's query
prompt1 = st.text_input("💬 What would you like to know from the document?")

# Q&A Execution
if prompt1 and "vectors" in st.session_state:
    with st.spinner("🤔 Thinking..."):
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        elapsed = round(time.process_time() - start, 4)

    st.write("🧠 **Answer:**", response['answer'])
    st.caption(f"⏱️ Response time: {elapsed} seconds")

    with st.expander("🔍 Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.markdown(doc.page_content)
            st.write("---")
