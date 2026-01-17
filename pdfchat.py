#WORKING#

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq  # Changed from Google Gemini to Groq
from langchain_core.prompts import ChatPromptTemplate
from pypdf import PdfReader
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up Streamlit page
st.set_page_config(page_title="PDF QA App", page_icon="📄", layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: #eef2f3;
        color: #2c3e50;
        font-family: Arial, sans-serif;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-size: 16px;
        padding: 8px 16px;
        border: none;
        border-radius: 6px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .stFileUploader {
        margin-bottom: 20px;
    }
    .stTextArea {
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Header
st.title("📄 PDF QA App with Generative AI")
st.write("Upload a PDF document, ask a question, and get accurate, detailed responses.")

# Get Groq API key from environment variable (loaded from .env file)
api_key = os.getenv("GROQ_API_KEY", "")

# Optional: Allow user to override with their own API key
if not api_key:
    api_key = st.text_input(
        "Enter your Groq API key:",
        type="password",
        help="Your API key is required to use Groq AI services. Get one free at: https://console.groq.com/keys",
    )
else:
    st.success("✅ Groq API Key loaded from environment")

if api_key:
    os.environ["GROQ_API_KEY"] = api_key

# File Upload Section
st.sidebar.header("📂 Upload a PDF")
uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF file to extract content:",
    type=["pdf"],
)

# User Question Section
st.sidebar.header("📝 Ask a Question")
user_question = st.sidebar.text_area(
    "Enter your question:",
    placeholder="E.g., What is the main topic of the document?",
)

@st.cache_data
def extract_text_from_pdf(pdf_file):
    """
    Extracts text from a PDF file using PyPDF2.
    """
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    if not text.strip():
        raise ValueError("No text could be extracted from the PDF. The PDF might be empty or contain only images.")
    return text

@st.cache_data
def get_text_chunks(text):
    """
    Splits the loaded text into chunks for embedding and retrieval.
    Optimized with larger chunks to reduce API calls.
    """
    if not text or not text.strip():
        raise ValueError("Text is empty, cannot create chunks.")
    # Larger chunks = fewer API calls (reduced from 10000 to save quota)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    if not chunks:
        raise ValueError("No text chunks were created.")
    # Limit to first 10 chunks to avoid quota exhaustion
    return chunks[:10]

@st.cache_resource
def get_vector_store(text_chunks):
    """
    Embeds the text chunks into a vector store for similarity search.
    Uses local embeddings - NO API CALLS, NO QUOTA LIMITS!
    """
    # Using local sentence-transformers model (runs on your computer)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain():
    """
    Creates a conversational chain for QA using LangChain and Groq.
    Using llama-3.3-70b-versatile (powerful, free tier friendly, fast)
    Other options: mixtral-8x7b-32768, llama-3.1-70b-versatile
    """
    model = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=2048
    )
    return model

def format_docs(docs):
    """Format documents for context"""
    return "\n\n".join([doc.page_content for doc in docs])

def answer_question(model, docs, question):
    """Generate answer from documents and question"""
    if not docs:
        return "No relevant documents found. Please make sure your PDF has extractable text."
    context = format_docs(docs)
    if not context.strip():
        return "No context available from the documents."
    prompt = f"""
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. 
    If the answer is not in the provided context, just say, "Answer is not available in the context." Don't provide a wrong answer.
    
    Context:\n{context}\n
    Question:\n{question}\n
    
    Answer:
    """
    response = model.invoke(prompt)
    return response.content

# Process Uploaded File
if uploaded_file and api_key and user_question:
    with st.spinner("Processing your PDF..."):
        try:
            # Step 1: Extract text from PDF
            st.info("📄 Extracting text from PDF...")
            raw_text = extract_text_from_pdf(uploaded_file)

            # Step 2: Split text into chunks
            # st.info(f"✂️ Splitting text into optimized chunks (larger chunks = fewer API calls)...")
            text_chunks = get_text_chunks(raw_text)
            st.info(f"📊 Created {len(text_chunks)} chunks from your document")

            # Step 3: Embed text into a vector store
            # st.info("🔄 Creating embeddings (using local AI - no API usage!)...")
            vector_store = get_vector_store(text_chunks)

            # Step 4: Perform similarity search and generate the answer
            st.info("🔍 Searching for relevant information (using local AI)...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            # Retrieve only top 2 most relevant documents to reduce processing
            docs = new_db.similarity_search(user_question, k=2)
            
            if not docs:
                st.warning("No relevant information found in the PDF for your question.")
                st.stop()
            
            model = get_conversational_chain()
            response_text = answer_question(model, docs, user_question)

            # Display the results
            st.success("Answer Generated!")
            st.subheader("Your Question:")
            st.write(user_question)

            st.subheader("Generated Answer:")
            for line in response_text.split("\n"):
                if line.strip():
                    st.write(line)
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    if not api_key:
        st.warning("Please provide your Google API key.")
    if not uploaded_file:
        st.warning("Please upload a PDF file.")
    if not user_question:
        st.warning("Please enter a question.")

# Footer
st.markdown(
    """
    ---
    🌟 Powered by LangChain, Groq AI (Llama 3.3), and Local Embeddings
    """
)