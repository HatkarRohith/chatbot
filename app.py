__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import tempfile
import base64
from pypdf import PdfReader
from fastembed import TextEmbedding
import chromadb
from chromadb.config import Settings
from groq import Groq

# Set page config
st.set_page_config(page_title="Multimodal RAG Agent", layout="wide", page_icon="🤖")

# --- AUTHENTICATION ---
api_key = st.secrets.get("GROQ_API_KEY")
if not api_key:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
    except:
        pass

if not api_key:
    with st.sidebar:
        api_key = st.text_input("Groq API Key", type="password")
    if not api_key:
        st.warning("Enter Groq API Key to continue.")
        st.stop()

# --- LOAD RESOURCES (Cached) ---
@st.cache_resource
def load_resources():
    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    # Use a persistent path for the DB so it doesn't reset on every rerun
    DB_DIR = os.path.join(tempfile.gettempdir(), "chroma_db_persistent")
    chroma_client = chromadb.PersistentClient(
        path=DB_DIR, 
        settings=Settings(anonymized_telemetry=False)
    )
    return embedder, chroma_client

client = Groq(api_key=api_key)
embedder, chroma_client = load_resources()

def get_collection():
    return chroma_client.get_or_create_collection(
        name="rag_collection",
        metadata={"hnsw:space": "cosine"}
    )

# --- HELPER: ENCODE IMAGE ---
def encode_image(uploaded_file):
    return base64.b64encode(uploaded_file.getvalue()).decode('utf-8')

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 Data & Vision")
    
    # 1. Document Upload (For RAG)
    st.markdown("### 📄 Knowledge Base")
    uploaded_files = st.file_uploader("Upload PDF/TXT", type=["pdf", "txt"], accept_multiple_files=True)
    process_btn = st.button("Process Documents")
    
    st.divider()

    # 2. Image Upload (For Multimodal / EdgeFleet)
    st.markdown("### 🖼️ Computer Vision")
    uploaded_image = st.file_uploader("Upload Image for Analysis", type=["jpg", "png", "jpeg"])
    analyze_btn = st.button("Analyze Image")

# --- PROCESS DOCUMENTS ---
if process_btn and uploaded_files:
    status = st.empty()
    status.info("Processing documents...")
    
    try:
        chroma_client.delete_collection("rag_collection")
    except:
        pass
    collection = get_collection()
    
    all_chunks = []
    for file in uploaded_files:
        text = ""
        try:
            if file.name.endswith(".pdf"):
                reader = PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() or ""
            elif file.name.endswith(".txt"):
                text = file.read().decode("utf-8")
            
            # Economy Chunking
            chunk_size = 800
            overlap = 100
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i : i + chunk_size]
                if len(chunk) > 50:
                    all_chunks.append(chunk)
        except:
            continue
            
    if all_chunks:
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            embeddings = [e.tolist() for e in list(embedder.embed(batch))]
            ids = [f"id_{i+j}" for j in range(len(batch))]
            collection.add(documents=batch, embeddings=embeddings, ids=ids)
        status.success(f"✅ Indexed {len(all_chunks)} chunks into Vector DB!")
    else:
        status.error("No text found in documents.")

# --- CHAT INTERFACE ---
st.title("🤖 Multimodal AI Agent")
st.caption("Powered by Groq LPU | RAG (Llama-3) + Vision (Llama-3.2)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg:
            with st.expander("🔍 Verified Sources (Explainability)"):
                for src in msg["sources"]:
                    st.info(src)

# --- IMAGE ANALYSIS LOGIC (UPDATED) ---
if analyze_btn and uploaded_image:
    with st.chat_message("user"):
        st.image(uploaded_image, caption="Analyzing this image...", width=300)
    
    with st.spinner("👀 AI is looking at your image..."):
        try:
            base64_image = encode_image(uploaded_image)
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in technical detail."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ],
                    }
                ],
                # ✅ CORRECT MODEL FOR VISION
                model="llama-3.2-11b-vision-preview", 
            )
            response_text = chat_completion.choices[0].message.content
            
            st.session_state.messages.append({"role": "user", "content": "Analyze uploaded image."})
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.rerun()
            
        except Exception as e:
            st.error(f"Error analyzing image: {e}")

# --- DOCUMENT Q&A LOGIC ---
if prompt := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    collection = get_collection()
    
    try:
        # Retrieve context
        q_embed = list(embedder.embed([prompt]))[0].tolist()
        results = collection.query(query_embeddings=[q_embed], n_results=5)
        
        source_docs = []
        context = ""
        
        if results['documents'] and results['documents'][0]:
            source_docs = results['documents'][0] 
            context = "\n".join(source_docs)
            
            if len(context) > 6000:
                context = context[:6000]
            
            sys_prompt = f"""
            You are a helpful AI assistant. Answer the question specifically using the context below.
            If the answer is not in the context, say "I cannot find this information in the documents."
            
            Context:
            {context}
            
            Question: {prompt}
            """
            
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": sys_prompt}],
                max_tokens=500
            )
            answer = response.choices[0].message.content
            
        else:
            answer = "I don't have any documents to answer from. Please upload a PDF first."

    except Exception as e:
        answer = f"Error: {str(e)}"
        source_docs = []

    # Append response WITH sources to history
    msg_data = {"role": "assistant", "content": answer}
    if source_docs:
        msg_data["sources"] = source_docs
        
    st.session_state.messages.append(msg_data)
    
    with st.chat_message("assistant"):
        st.write(answer)
        if source_docs:
            with st.expander("🔍 Verified Sources (Explainability)"):
                for src in source_docs:
                    st.info(src[:300] + "...")
