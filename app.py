__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv

import tempfile
import time
import openai
import logging

# Load environment variables
load_dotenv()

# Get OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit app title and description
st.title("Chat with PDFs using RAG with LangChain")
st.write("Upload a PDF file to chat and ask questions about its content.")

if openai_api_key:
    # Upload PDF file
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Save uploaded PDF to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Load PDF using PyPDFLoader
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()

        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        try:
            # Create ChromaDB vector store
            vector_store = Chroma(embedding_function=embeddings)
            logger.info("Chroma vector store initialized successfully.")
            
            # Add documents to the vector store with retry logic
            def add_documents_with_retry(vector_store, documents, retries=3):
                for attempt in range(retries):
                    try:
                        vector_store.add_documents(documents)
                        return
                    except openai.error.RateLimitError:
                        if attempt < retries - 1:
                            st.warning("Rate limit exceeded, retrying...")
                            time.sleep(10)  # Wait for 10 seconds before retrying
                        else:
                            st.error("Exceeded rate limit. Please check your OpenAI plan and billing details.")
                            return
            
            add_documents_with_retry(vector_store, documents)

            # Initialize the Retriever
            retriever = vector_store.as_retriever()

            # Initialize the Chat Model
            llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4-turbo")

            # Create the Conversational Retrieval Chain
            qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)

            # Initialize chat history
            chat_history = []

            # Question input
            question = st.text_input("Ask a question about the PDF")

            if question:
                # Get the answer from the QA chain
                result = qa_chain({"question": question, "chat_history": chat_history})
                st.write("Answer:", result["answer"])

                # Update chat history
                chat_history.append({"question": question, "answer": result["answer"]})

        except Exception as e:
            logger.error(f"Failed to initialize Chroma vector store: {e}")
            st.error(f"Failed to initialize Chroma vector store: {e}")
else:
    st.error("Please set the OpenAI API key in the .env file.")
