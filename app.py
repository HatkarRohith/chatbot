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
st.set_page_config(page_title="Multimodal AI Agent", layout="wide", page_icon="🤖")

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

def encode_image(uploaded_file):
    return base64.b64encode(uploaded_file.getvalue()).decode('utf-8')

# --- SIDEBAR: UNIVERSAL UPLOADER ---
with st.sidebar:
    st.header("📂 Data Input")
    st.info("Upload any file type here. The AI will sort them automatically!")
    
    # 1. Single Universal Uploader
    uploaded_files = st.file_uploader(
        "Upload Documents or Images", 
        type=["pdf", "txt", "png", "jpg", "jpeg"], 
        accept_multiple_files=True
    )

    # 2. Logic to Split Files
    docs_to_process = []
    images_to_process = []
    
    if uploaded_files:
        for file in uploaded_files:
            if file.name.endswith(('.pdf', '.txt')):
                docs_to_process.append(file)
            elif file.name.endswith(('.png', '.jpg', '.jpeg')):
                images_to_process.append(file)
        
        st.divider()
        
        # Display Status
        if docs_to_process:
            st.markdown(f"**📄 Documents:** {len(docs_to_process)} detected")
            process_btn = st.button("🧠 Process Text Data", type="primary")
        
        if images_to_process:
            st.markdown(f"**🖼️ Images:** {len(images_to_process)} detected")
            # We don't need a button for images, we handle them in the chat flow
            st.caption("Images are ready for analysis in the chat!")

# --- PROCESSING LOGIC (DOCS) ---
if uploaded_files and docs_to_process and 'process_btn' in locals() and process_btn:
    status = st.empty()
    status.info("Processing documents...")
    
    try:
        chroma_client.delete_collection("rag_collection")
    except:
        pass
    collection = get_collection()
    
    all_chunks = []
    for file in docs_to_process:
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
        status.success(f"✅ Indexed {len(all_chunks)} chunks!")
    else:
        status.error("No text found.")

# --- CHAT INTERFACE ---
st.title("🤖 Universal AI Agent")
st.caption("Powered by Groq LPU | RAG (Llama-3) + Vision (Llama-3.2-90b)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg:
            with st.expander("🔍 Verified Sources"):
                for src in msg["sources"]:
                    st.info(src)
        if "image" in msg:
            st.image(msg["image"], width=250)

# --- IMAGE ANALYSIS (AUTO TRIGGER) ---
# If an image was just uploaded and hasn't been analyzed, let the user trigger it
if images_to_process:
    current_image = images_to_process[-1] # Take the most recent one
    
    with st.chat_message("assistant"):
        st.write("I see you uploaded an image. What would you like to know?")
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image(current_image, width=150, caption=current_image.name)
        with col2:
            if st.button("👀 Describe this Image"):
                with st.spinner("Analyzing..."):
                    try:
                        base64_image = encode_image(current_image)
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
                            # ✅ UPDATED MODEL: 90b is often more stable than 11b-preview
                            model="llama-3.2-90b-vision-preview", 
                        )
                        response_text = chat_completion.choices[0].message.content
                        
                        # Add to history
                        st.session_state.messages.append({"role": "user", "content": f"Analyze image: {current_image.name}"})
                        st.session_state.messages.append({"role": "assistant", "content": response_text, "image": current_image})
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

# --- DOCUMENT Q&A LOGIC ---
if prompt := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    collection = get_collection()
    
    try:
        q_embed = list(embedder.embed([prompt]))[0].tolist()
        results = collection.query(query_embeddings=[q_embed], n_results=5)
        
        source_docs = []
        context = ""
        
        if results['documents'] and results['documents'][0]:
            source_docs = results['documents'][0] 
            context = "\n".join(source_docs)
            if len(context) > 6000: context = context[:6000]
            
            sys_prompt = f"""
            You are a helpful AI assistant. Answer using the context below.
            Context: {context}
            Question: {prompt}
            """
            
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": sys_prompt}],
                max_tokens=500
            )
            answer = response.choices[0].message.content
        else:
            # Fallback if no docs found - just chat normally
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            answer = response.choices[0].message.content
            source_docs = []

    except Exception as e:
        answer = f"Error: {str(e)}"
        source_docs = []

    msg_data = {"role": "assistant", "content": answer}
    if source_docs: msg_data["sources"] = source_docs
        
    st.session_state.messages.append(msg_data)
    with st.chat_message("assistant"):
        st.write(answer)
        if source_docs:
            with st.expander("🔍 Sources"):
                for src in source_docs: st.info(src[:300] + "...")
