'''working with Groq Llama 3.3 and Local Embeddings to build an interactive QA app using Streamlit'''

import os
# Set USER_AGENT FIRST before any other imports that might check it
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Only try this if Step 1 doesn't work for your version
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq  # Changed from Google Gemini to Groq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up page configuration
st.set_page_config(page_title="Interactive QA App", page_icon="🧙‍♂️", layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: #f0f4f5;
        color: #333333;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px;
        margin: 10px;
        border-radius: 8px;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Header
st.title("📚 Interactive QA App with Generative AI")
st.write("Ask detailed questions based on contextual data, and get accurate and rich responses.")

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

# Load Data
@st.cache_data
def get_url_text(url):
    """
    Loads content from a predefined URL and processes it into a string.
    Detects and blocks JavaScript-protected websites.
    """
    if not url or not url.strip():
        raise ValueError("URL cannot be empty.")
    loader = WebBaseLoader(url)
    documents = loader.load()
    if not documents:
        raise ValueError("No content could be loaded from the URL.")
    text = "\n\n".join([doc.page_content for doc in documents])
    if not text.strip():
        raise ValueError("No text content found in the URL.")
    
    # Detect if website is blocking scraping with JavaScript/Cloudflare protection
    blocking_phrases = [
        "Just a moment",
        "Enable JavaScript and cookies to continue",
        "Checking your browser",
        "Please enable JavaScript",
        "Cloudflare",
        "Access denied",
        "Security check"
    ]
    
    text_lower = text.lower()
    is_blocked = any(phrase.lower() in text_lower for phrase in blocking_phrases)
    
    # Check if content is suspiciously short (less than 500 characters)
    if len(text) < 500 or is_blocked:
        raise ValueError(
            f"⚠️ Website is blocking scraping (detected JavaScript/cookies requirement).\n\n"
            f"Content length: {len(text)} characters\n\n"
            f"This website requires JavaScript or has anti-bot protection.\n\n"
            f"✅ Please try these compatible websites instead:\n"
            f"• Wikipedia: https://en.wikipedia.org/\n"
            f"• Medium articles: https://medium.com/\n"
            f"• News sites: https://www.bbc.com/news\n"
            f"• Documentation sites: https://docs.python.org/\n"
            f"• Blogs without heavy JavaScript protection\n\n"
            f"❌ Avoid: Sites with Cloudflare, heavy JavaScript, or login requirements"
        )
    
    return text


@st.cache_data
def get_text_chunks(text):
    """
    Splits the loaded text into chunks for embedding and retrieval.
    Optimized with larger chunks to reduce API calls.
    """
    if not text or not text.strip():
        raise ValueError("Text is empty, cannot create chunks.")
    # Larger chunks = fewer API calls
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=2000)
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

st.sidebar.header("📝 Enter the URL")
st.sidebar.write("Provide a URL and ask a question based on the document's content.")

# Add helpful guidance about compatible websites
with st.sidebar.expander("ℹ️ Website Compatibility Guide"):
    st.markdown("""
    **✅ Compatible Websites:**
    - Wikipedia articles
    - Medium.com articles  
    - News sites (BBC, Reuters)
    - Documentation sites
    - Simple blogs
    
    **❌ Incompatible Websites:**
    - Sites with Cloudflare protection
    - JavaScript-heavy sites
    - Sites requiring login
    - Dynamic content sites
    
    **Common Error:** If you see "Just a moment..." or very short content (< 500 chars), 
    the website is blocking scraping. Try a different URL!
    """)

# Input for URL
url = st.sidebar.text_input("Enter the URL to extract context from:", 
    placeholder="e.g., https://en.wikipedia.org/wiki/Harry_Potter")


# Prepare Data
if api_key:
    st.sidebar.header("📋 Preparing Data...")

# Conversational Chain
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
        return "No relevant documents found. Please make sure the URL has extractable text."
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




# User Interaction
st.sidebar.header("📝 Ask a Question")
user_question = st.sidebar.text_area(
    "Enter your question:",
    placeholder="E.g., Name and Plot of Harry Potter First Movie.",
)

if st.sidebar.button("Get Answer"):
    if url and api_key and user_question:
        try:
            with st.spinner("Searching for the answer..."):
                st.info("🌐 Loading content from URL...")
                text = get_url_text(url)
                
                # Show content length to verify successful scraping
                st.success(f"✅ Successfully loaded {len(text)} characters from URL")
                
                # st.info("✂️ Splitting text into optimized chunks (larger chunks = fewer API calls)...")
                chunks = get_text_chunks(text)
                st.info(f"📊 Created {len(chunks)} chunks from the URL content")
                
                # Display what's being stored in each chunk
                st.subheader("📦 Chunk Preview - What's Getting Stored:")
                for i, chunk in enumerate(chunks, 1):
                    with st.expander(f"Chunk {i} - {len(chunk)} characters"):
                        st.write(f"**Preview :**")
                        st.code(chunk[:500] + "..." if len(chunk) > 500 else chunk, language="text")
                
                # st.info("🔄 Creating embeddings (using local AI - no API usage!)...")
                vectorstore = get_vector_store(chunks)
                
                # st.info("🔍 Searching for relevant information (using local AI)...")
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
                new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                # Retrieve only top 2 most relevant documents to reduce processing
                docs = new_db.similarity_search(user_question, k=2)
                
                if not docs:
                    st.warning("No relevant information found for your question.")
                    st.stop()
                
                st.info("🤖 Generating answer with Groq (Llama 3.3)...")
                model = get_conversational_chain()
                response_text = answer_question(model, docs, user_question)

            st.success("Answer Generated!")
            st.subheader("Your Question:")
            st.write(user_question)

            st.subheader("Generated Answer:")
            # Display the response line by line
            for line in response_text.split("\n"):
                if line.strip(): 
                    st.write(line)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        if not url:
            st.warning("Please provide a URL.")
        if not api_key:
            st.warning("Please provide your Google API key.")
        if not user_question:
            st.warning("Please enter a question.")

# Footer
st.markdown(
    """
    ---
    🤖 Powered by LangChain, Groq AI (Llama 3.3), and Local Embeddings
    """
)
