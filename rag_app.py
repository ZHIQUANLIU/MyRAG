"""
RAG (Retrieval Augmented Generation) Application
Using LangChain + Google Gemini
"""

import os
import time
import sys
from pathlib import Path
from datetime import datetime
from collections import deque

from dotenv import load_dotenv

class RollingLogger:
    def __init__(self, log_file="rag_app.log", max_bytes=20*1024*1024, backup_count=3):
        self.log_file = log_file
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.terminal = sys.stdout
        self._open()
    
    def _open(self):
        self.log = open(self.log_file, "a", encoding="utf-8")
    
    def _rotate(self):
        self.log.close()
        for i in range(self.backup_count - 1, 0, -1):
            src = f"{self.log_file}.{i}"
            dst = f"{self.log_file}.{i+1}"
            if os.path.exists(src):
                if os.path.exists(dst):
                    os.remove(dst)
                os.rename(src, dst)
        if os.path.exists(self.log_file):
            os.rename(self.log_file, f"{self.log_file}.1")
        self._open()
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        if os.path.getsize(self.log_file) > self.max_bytes:
            self._rotate()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

logger = RollingLogger("rag_app.log", max_bytes=20*1024*1024, backup_count=3)
sys.stdout = logger

print(f"=== RAG App Log {datetime.now()} ===")

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredExcelLoader, UnstructuredEPubLoader
)
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY in .env file")

PDF_FOLDER = "my_pdfs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3
VECTOR_STORE_PATH = "faiss_index"

EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash-lite"


def load_documents_from_folder(folder_path: str):
    """Load all supported documents from folder (PDF, TXT, DOCX, XLSX, EPUB)"""
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Folder {folder_path} does not exist, created")
        folder.mkdir(parents=True, exist_ok=True)
        return []
    
    extensions = {
        ".pdf": "PDF",
        ".txt": "TXT",
        ".docx": "DOCX",
        ".doc": "DOCX",
        ".xlsx": "XLSX",
        ".xls": "XLSX",
        ".epub": "EPUB"
    }
    
    files_by_type = {ext: [] for ext in extensions}
    for ext, _ in extensions.items():
        files_by_type[ext] = list(folder.glob(f"*{ext}"))
    
    total_files = sum(len(f) for f in files_by_type.values())
    
    if total_files == 0:
        print(f"No supported files found in {folder_path}")
        print("Supported formats: PDF, TXT, DOCX, XLSX, EPUB")
        return []
    
    print(f"\nFound {total_files} files:")
    for ext, files in files_by_type.items():
        if files:
            print(f"   {extensions[ext]}: {len(files)} files - {', '.join([f.name for f in files[:3]])}{'...' if len(files) > 3 else ''}")
    
    documents = []
    total_pages = 0
    
    for ext, files in files_by_type.items():
        for file_path in files:
            print(f"\nLoading: {file_path.name}")
            try:
                doc = None
                if ext == ".pdf":
                    from pypdf import PdfReader
                    reader = PdfReader(str(file_path))
                    print(f"   PDF has {len(reader.pages)} pages")
                    for i, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text and text.strip():
                            doc = Document(
                                page_content=text,
                                metadata={"source": file_path.name, "page": i+1}
                            )
                            documents.append(doc)
                    print(f"   Extracted {len(reader.pages)} pages")
                    
                elif ext == ".txt":
                    loader = TextLoader(str(file_path), encoding="utf-8")
                    docs = loader.load()
                    print(f"   Loaded {len(docs)} document(s)")
                    documents.extend(docs)
                    
                elif ext in [".docx", ".doc"]:
                    loader = Docx2txtLoader(str(file_path))
                    docs = loader.load()
                    print(f"   Loaded {len(docs)} document(s)")
                    documents.extend(docs)
                    
                elif ext in [".xlsx", ".xls"]:
                    loader = UnstructuredExcelLoader(str(file_path))
                    docs = loader.load()
                    print(f"   Loaded {len(docs)} document(s)")
                    documents.extend(docs)
                    
                elif ext == ".epub":
                    loader = UnstructuredEPubLoader(str(file_path))
                    docs = loader.load()
                    print(f"   Loaded {len(docs)} document(s)")
                    documents.extend(docs)
                    
            except Exception as e:
                print(f"   Failed: {str(e)}")
                import traceback
                traceback.print_exc()
    
    print(f"\nTotal: {len(documents)} documents loaded")
    
    if documents:
        print(f"First doc content length: {len(documents[0].page_content)}")
        print(f"First doc preview: {documents[0].page_content[:200]}")
    
    return documents


def split_documents(documents: list, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    if not documents:
        print("No documents to split")
        return []
    
    print(f"\nSplitting documents (chunk_size={chunk_size}, overlap={chunk_overlap})...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True
    )
    
    splits = text_splitter.split_documents(documents)
    
    print(f"   Split complete, {len(splits)} text chunks created")
    if splits:
        print(f"   Sample content length: {len(splits[0].page_content)}")
    return splits


def create_or_load_vectorstore(splits: list, persist_path: str = VECTOR_STORE_PATH):
    if os.path.exists(persist_path) and os.path.exists(os.path.join(persist_path, "index.faiss")):
        print(f"\nFound existing vector database, loading...")
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model=EMBEDDING_MODEL,
                google_api_key=GOOGLE_API_KEY
            )
            vectorstore = FAISS.load_local(
                persist_path, 
                embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"   Vector database loaded successfully!")
            return vectorstore
        except Exception as e:
            print(f"   Load failed: {str(e)}, recreating")
    
    if not splits:
        print("No text chunks to create vector database")
        return None
    
    print(f"\nCreating vector database (using {EMBEDDING_MODEL})...")
    print("   This may take a few minutes, please wait...\n")
    
    start_time = time.time()
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY
        )
        
        vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=embeddings
        )
        
        print(f"\nSaving vector database to {persist_path}...")
        vectorstore.save_local(persist_path)
        
        elapsed_time = time.time() - start_time
        print(f"   Vector database created! (Time: {elapsed_time:.2f}s)")
        
        return vectorstore
        
    except Exception as e:
        print(f"   Creation failed: {str(e)}")
        raise


def create_qa_chain(vectorstore):
    if not vectorstore:
        raise ValueError("Vector database is empty, cannot create QA chain")
    
    print("\nInitializing LLM and QA chain...")
    
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7,
        convert_system_message_to_human=True
    )
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )
    
    prompt_template = """You are a professional knowledge Q&A assistant. Please answer based on the following context.

Context:
{context}

Question: {question}

Please provide a clear and accurate answer. If the context does not contain relevant information, please inform the user."""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    print(f"   LLM model: {LLM_MODEL}")
    print(f"   Retrieval top_k: {TOP_K}")
    
    return qa_chain


def interactive_qa(qa_chain):
    print("\n" + "="*60)
    print("RAG Interactive QA System Ready")
    print("="*60)
    print("Enter question to query, type 'quit' or 'exit' to exit")
    print("Type 'sources' to view retrieved source documents")
    print("="*60 + "\n")
    
    last_result = None
    
    while True:
        try:
            question = input("Please enter question: ").strip()
            
            if not question:
                print("Please enter a valid question")
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if question.lower() == 'sources':
                if last_result and 'source_documents' in last_result:
                    print("\nRetrieved source documents:")
                    print("-"*40)
                    for i, doc in enumerate(last_result['source_documents'], 1):
                        print(f"\n[Document {i}]")
                        content = doc.page_content[:300]
                        print(content + "..." if len(doc.page_content) > 300 else content)
                    print("-"*40)
                else:
                    print("No source documents from last query")
                continue
            
            print("\nThinking...\n")
            
            result = qa_chain({"query": question})
            last_result = result
            
            print("Answer:")
            print("-"*40)
            print(result['result'])
            print("-"*40 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nExited")
            break
        except Exception as e:
            print(f"\nError: {str(e)}\n")


def main():
    print("\n" + "="*60)
    print("RAG Application Started")
    print("="*60)
    print(f"PDF Folder: {PDF_FOLDER}")
    print(f"Chunk Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
    print(f"Top K: {TOP_K}")
    print(f"Embedding: {EMBEDDING_MODEL}")
    print(f"LLM: {LLM_MODEL}")
    print("="*60 + "\n")
    
    documents = load_documents_from_folder(PDF_FOLDER)
    
    if not documents:
        print("No PDF documents found. Please put PDF files in my_pdfs folder")
        return
    
    splits = split_documents(documents)
    
    if not splits:
        print("Document splitting failed")
        return
    
    vectorstore = create_or_load_vectorstore(splits)
    
    if not vectorstore:
        print("Vector database creation failed")
        return
    
    qa_chain = create_qa_chain(vectorstore)
    
    interactive_qa(qa_chain)


if __name__ == "__main__":
    main()