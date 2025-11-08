"""
App 1: Document Q&A with Local LLM and RAG
===========================================

A simple document question-answering system that:
1. Ingests a PDF document
2. Chunks and embeds the text locally
3. Stores embeddings in ChromaDB
4. Answers questions using retrieved context + Ollama LLM

All processing happens locally - your documents never leave your machine.
"""

import os
import streamlit as st
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pypdf
import requests
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

# Where to store the ChromaDB database (change this to your preferred location)
CHROMA_DB_PATH = "./chroma_db"

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"

# Chunking configuration
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 50  # character overlap between chunks

# Retrieval configuration
TOP_K = 3  # number of chunks to retrieve for each question

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_embedding_model():
    """Load the sentence transformer model for embeddings (cached)."""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_chroma_client():
    """Initialize ChromaDB client (cached)."""
    # Create the directory if it doesn't exist
    Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
    
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client

def extract_text_from_pdf(pdf_file):
    """
    Extract text from uploaded PDF file.
    
    Args:
        pdf_file: Streamlit UploadedFile object
        
    Returns:
        str: Extracted text from all pages
    """
    pdf_reader = pypdf.PdfReader(pdf_file)
    text = ""
    
    for page_num, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text()
        text += f"\n--- Page {page_num + 1} ---\n{page_text}"
    
    return text

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping chunks.
    
    Simple fixed-size chunking with character overlap.
    Later modules will explore smarter chunking strategies.
    
    Args:
        text: Full document text
        chunk_size: Size of each chunk in characters
        overlap: Number of overlapping characters between chunks
        
    Returns:
        list: List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
        
        start += chunk_size - overlap
    
    return chunks

def create_embeddings(chunks, model):
    """
    Generate embeddings for text chunks.
    
    Args:
        chunks: List of text chunks
        model: SentenceTransformer model
        
    Returns:
        list: List of embedding vectors
    """
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings.tolist()

def store_in_chromadb(chunks, embeddings, metadata, collection_name="documents"):
    """
    Store chunks and embeddings in ChromaDB.
    
    Args:
        chunks: List of text chunks
        embeddings: List of embedding vectors
        metadata: List of metadata dicts for each chunk
        collection_name: Name of the ChromaDB collection
    """
    client = get_chroma_client()
    
    # Delete existing collection if it exists (fresh start for each upload)
    try:
        client.delete_collection(name=collection_name)
    except:
        pass
    
    # Create new collection
    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "Document chunks for Q&A"}
    )
    
    # Add documents to collection
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    collection.add(
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadata,
        ids=ids
    )
    
    return collection

def retrieve_relevant_chunks(question, collection, model, top_k=TOP_K):
    """
    Retrieve most relevant chunks for a question.
    
    Args:
        question: User's question
        collection: ChromaDB collection
        model: SentenceTransformer model
        top_k: Number of chunks to retrieve
        
    Returns:
        dict: Retrieved chunks with metadata and distances
    """
    # Generate embedding for the question
    question_embedding = model.encode([question])[0].tolist()
    
    # Query the collection
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )
    
    return results

def query_ollama(prompt, model=OLLAMA_MODEL):
    """
    Send a prompt to Ollama and get response.
    
    Args:
        prompt: The prompt to send
        model: Ollama model name
        
    Returns:
        str: Generated response
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        return f"Error connecting to Ollama: {str(e)}\n\nMake sure Ollama is running with: ollama serve"

def generate_answer(question, retrieved_chunks):
    """
    Generate answer using retrieved context and Ollama.
    
    Args:
        question: User's question
        retrieved_chunks: Retrieved context from ChromaDB
        
    Returns:
        tuple: (answer, sources)
    """
    # Extract the text chunks and metadata
    chunks = retrieved_chunks['documents'][0]
    metadatas = retrieved_chunks['metadatas'][0]
    
    # Build context from retrieved chunks
    context = "\n\n".join([f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)])
    
    # Create prompt for the LLM
    prompt = f"""You are a helpful assistant answering questions based on provided document context.

Context from the document:
{context}

Question: {question}

Instructions:
- Answer the question using ONLY the information provided in the context above
- If the context doesn't contain relevant information, say "I don't have information about that in the provided document"
- Be concise and direct
- Cite which chunk number(s) you used to formulate your answer

Answer:"""
    
    # Get answer from Ollama
    answer = query_ollama(prompt)
    
    # Format sources
    sources = []
    for i, metadata in enumerate(metadatas):
        sources.append({
            "chunk_num": i + 1,
            "page": metadata.get("page", "unknown"),
            "text": chunks[i][:200] + "..." if len(chunks[i]) > 200 else chunks[i]
        })
    
    return answer, sources

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="wide")
    
    st.title("📄 Document Q&A with Local LLM")
    st.markdown("*Privacy-first: All processing happens on your machine*")
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📤 Upload Document")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF document to ask questions about"
        )
        
        if uploaded_file:
            st.success(f"Loaded: {uploaded_file.name}")
            
            if st.button("Process Document", type="primary"):
                with st.spinner("Processing document..."):
                    try:
                        # Extract text
                        st.info("Extracting text from PDF...")
                        text = extract_text_from_pdf(uploaded_file)
                        
                        # Chunk text
                        st.info("Chunking text...")
                        chunks = chunk_text(text)
                        
                        # Load embedding model
                        st.info("Loading embedding model...")
                        model = load_embedding_model()
                        
                        # Generate embeddings
                        st.info(f"Generating embeddings for {len(chunks)} chunks...")
                        embeddings = create_embeddings(chunks, model)
                        
                        # Create metadata for each chunk
                        metadata = [{"filename": uploaded_file.name, "chunk_index": i} for i in range(len(chunks))]
                        
                        # Store in ChromaDB
                        st.info("Storing in vector database...")
                        collection = store_in_chromadb(chunks, embeddings, metadata)
                        
                        # Store in session state
                        st.session_state['document_processed'] = True
                        st.session_state['collection'] = collection
                        st.session_state['model'] = model
                        st.session_state['filename'] = uploaded_file.name
                        st.session_state['num_chunks'] = len(chunks)
                        
                        st.success("✅ Document processed successfully!")
                        st.info(f"Created {len(chunks)} chunks stored in: `{CHROMA_DB_PATH}`")
                        
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
        
        # Display document info if processed
        if st.session_state.get('document_processed'):
            st.markdown("---")
            st.markdown("### 📊 Document Info")
            st.markdown(f"**File:** {st.session_state.get('filename', 'N/A')}")
            st.markdown(f"**Chunks:** {st.session_state.get('num_chunks', 0)}")
            st.markdown(f"**DB Path:** `{CHROMA_DB_PATH}`")
    
    # Main area for Q&A - ALWAYS VISIBLE
    st.header("💬 Ask Questions")
    
    # Show current mode
    if st.session_state.get('document_processed'):
        st.success("✅ **RAG Mode:** Answers will be based on your uploaded document")
    else:
        st.warning("⚠️ **Direct LLM Mode:** No document loaded - answers come from the LLM's training data only")
        st.info("💡 **Try this:** Ask a question now, then upload a document and ask the same question to see the difference!")
    
    question = st.text_input(
        "Enter your question:",
        placeholder="What is retrieval-augmented generation?" if not st.session_state.get('document_processed') else "What is this document about?",
        help="Ask any question - behavior changes based on whether a document is loaded"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("Ask", type="primary")
    
    if ask_button and question:
        with st.spinner("Thinking..."):
            try:
                if st.session_state.get('document_processed'):
                    # RAG MODE: Retrieve relevant chunks and generate answer
                    collection = st.session_state['collection']
                    model = st.session_state['model']
                    
                    retrieved = retrieve_relevant_chunks(question, collection, model)
                    answer, sources = generate_answer(question, retrieved)
                    
                    # Display answer
                    st.markdown("### 🤖 Answer (from document)")
                    st.markdown(answer)
                    
                    # Display sources
                    st.markdown("### 📚 Sources Used")
                    for source in sources:
                        with st.expander(f"Chunk {source['chunk_num']} (Page: {source['page']})"):
                            st.text(source['text'])
                else:
                    # DIRECT LLM MODE: No retrieval, just ask the LLM directly
                    prompt = f"""You are a helpful assistant. Answer the following question directly based on your training data.

Question: {question}

Answer:"""
                    
                    answer = query_ollama(prompt)
                    
                    # Display answer
                    st.markdown("### 🤖 Answer (from LLM training data)")
                    st.markdown(answer)
                    st.info("💡 This answer came from the LLM's training data, not from any specific document. Upload a document to see how RAG provides more accurate, grounded answers.")
                
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
    
    elif ask_button:
        st.warning("Please enter a question.")
    
    # Instructions section at the bottom
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        ### Two Modes:
        
        **Direct LLM Mode** (no document):
        - Questions go straight to the language model
        - Answers come from the model's training data
        - May hallucinate or give generic answers
        
        **RAG Mode** (document uploaded):
        1. Your question is converted to an embedding
        2. System retrieves most relevant chunks from your document
        3. Retrieved chunks are sent to the LLM as context
        4. LLM answers based on YOUR document, not training data
        
        ### See the Difference:
        1. Ask a question about a specific topic
        2. Note the generic answer
        3. Upload a document about that topic
        4. Ask the same question again
        5. Compare the answers!
        
        All processing happens locally on your machine - your documents never leave your computer.
        """)
        
        st.markdown("### Requirements:")
        st.code("""
# Make sure Ollama is running:
ollama serve

# And the model is pulled:
ollama pull llama3.2:3b
        """)

if __name__ == "__main__":
    # Initialize session state
    if 'document_processed' not in st.session_state:
        st.session_state['document_processed'] = False
    
    main()