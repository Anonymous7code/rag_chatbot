import os
import tempfile
import streamlit as st

# 1. Partner Packages (Latest standard for Ollama)
from langchain_ollama import ChatOllama, OllamaEmbeddings

# 2. Community Packages (For Loaders and Vector Stores)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# 3. Core Packages (For Text Splitting and Prompts)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# 4. Chains (Modern LCEL Architecture)
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Legal Contract AI Assistant", page_icon="⚖️", layout="wide")
st.title("⚖️ Legal Contract Chatbot (Latest Architecture)")
st.subheader("Upload a contract, ask questions, and get answers with citations.")

# Initialize Session States
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar: Document Upload & Processing ---
with st.sidebar:
    st.header("1. Upload Contract")
    uploaded_file = st.file_uploader("Choose a PDF contract", type=["pdf"])
    
    if uploaded_file is not None and st.session_state.vector_store is None:
        with st.spinner("Processing contract... This might take a moment."):
            # Safely handle the uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # 1. Load PDF
                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load()
                
                # 2. Split Text (Optimized for legal context)
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=200,
                    separators=["\n\n", "\n", ".", " "]
                )
                split_docs = text_splitter.split_documents(docs)
                
                # 3. Initialize Modern Ollama Embeddings
                embeddings = OllamaEmbeddings(model="llama3.2")
                
                # 4. Create FAISS Vector DB
                st.session_state.vector_store = FAISS.from_documents(split_docs, embeddings)
                st.success("Contract successfully indexed! You can now ask questions.")
                
            except Exception as e:
                st.error(f"Error processing file: {e}")
            finally:
                # Cleanup temp file
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)

# --- Main Chat Interface ---
if st.session_state.vector_store is not None:
    # Render existing chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "citations" in message and message["citations"]:
                with st.expander("View Citations 📄"):
                    for citation in message["citations"]:
                        st.caption(citation)

    # User Input
    if user_query := st.chat_input("Ask something about the contract (e.g., 'What is the termination clause?')"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing contract clauses..."):
                # 1. Setup LLM (Using the dedicated langchain-ollama package)
                llm = ChatOllama(model="llama3.2", temperature=0.1) 
                
                # 2. Setup Retriever
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
                
                # 3. Create LCEL System Prompt
                # The context placeholder {context} is automatically populated by create_stuff_documents_chain
                system_prompt = (
                    "You are a meticulous legal assistant. Analyze the provided contract context "
                    "to answer the user's question. If you do not know the answer based on the context, "
                    "state clearly that the information is not present in the document. Do not make things up.\n\n"
                    "Context:\n{context}"
                )
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])
                
                # 4. Build Modern LangChain QA & Retrieval Chain (LCEL)
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                
                # 5. Invoke Chain
                response = rag_chain.invoke({"input": user_query})
                
                answer = response["answer"]
                source_documents = response.get("context", [])
                
                # Format Citations from Metadata
                citations = []
                seen_sources = set()
                for doc in source_documents:
                    page = doc.metadata.get("page", 0) + 1  # PyPDFLoader is 0-indexed
                    # Clean up the snippet for display
                    snippet = doc.page_content[:150].strip().replace('\n', ' ')
                    citation_str = f"**Page {page}**: \"...{snippet}...\""
                    
                    if citation_str not in seen_sources:
                        citations.append(citation_str)
                        seen_sources.add(citation_str)
                
                # Render Answer
                st.markdown(answer)
                if citations:
                    with st.expander("View Citations 📄"):
                        for citation in citations:
                            st.caption(citation)
                            
                # Save to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations
                })
else:
    st.info("👈 Please upload a legal contract PDF in the sidebar to begin.")