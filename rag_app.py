"""
RAG Application with GUI
Using LangChain + Google Gemini + Tkinter
"""

import os
import time
import sys
import threading
import queue
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

class RollingLogger:
    def __init__(self, log_file="rag_app.log", max_bytes=20*1024*1024, backup_count=3):
        self.log_file = log_file
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.terminal = sys.stdout
        self.log_queue = queue.Queue()
        self.show_log = True
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
        self.terminal.write(message if self.show_log else "")
        self.log.write(message)
        self.log.flush()
        try:
            if os.path.getsize(self.log_file) > self.max_bytes:
                self._rotate()
        except:
            pass
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

logger = RollingLogger("rag_app.log", max_bytes=20*1024*1024, backup_count=3)
sys.stdout = logger

print(f"=== RAG App Log {datetime.now()} ===")

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader, Docx2txtLoader, UnstructuredExcelLoader, UnstructuredEPubLoader
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


class RAGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RAG Application")
        self.root.geometry("900x700")
        
        self.vectorstore = None
        self.qa_chain = None
        self.embedded_files = []
        self.embedding_thread = None
        self.stop_embedding = False
        
        self.setup_ui()
        self.load_vectorstore()
    
    def setup_ui(self):
        # Top frame - File management
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.BOTH, expand=False)
        
        # Embedded files section
        ttk.Label(top_frame, text="Embedded Files:", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        self.files_listbox = tk.Listbox(top_frame, height=8, width=80)
        self.files_listbox.pack(fill=tk.BOTH, pady=5)
        scrollbar = ttk.Scrollbar(top_frame, orient="vertical", command=self.files_listbox.yview)
        self.files_listbox.configure(yscrollcommand=scrollbar.set)
        
        # Button frame
        btn_frame = ttk.Frame(top_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="Add Files", command=self.add_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Re-embed All", command=self.reembed_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Clear Database", command=self.clear_database).pack(side=tk.LEFT, padx=5)
        
        # Log frame
        log_frame = ttk.LabelFrame(top_frame, text="Embedding Log", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, width=80, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Toggle log button
        self.log_visible = tk.BooleanVar(value=True)
        ttk.Checkbutton(log_frame, text="Show Log", variable=self.log_visible, 
                       command=self.toggle_log).pack(anchor=tk.W)
        
        # Chat frame
        chat_frame = ttk.LabelFrame(self.root, text="Q&A Chat", padding="10")
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Answer display
        ttk.Label(chat_frame, text="Answer:").pack(anchor=tk.W)
        self.answer_text = scrolledtext.ScrolledText(chat_frame, height=12, width=80)
        self.answer_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Question input
        input_frame = ttk.Frame(chat_frame)
        input_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(input_frame, text="Question:").pack(side=tk.LEFT)
        self.question_entry = ttk.Entry(input_frame, width=60)
        self.question_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.question_entry.bind("<Return>", self.ask_question)
        
        ttk.Button(input_frame, text="Ask", command=self.ask_question).pack(side=tk.LEFT)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X)
    
    def log(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        print(message)
    
    def toggle_log(self):
        logger.show_log = self.log_visible.get()
    
    def load_vectorstore(self):
        if os.path.exists(VECTOR_STORE_PATH) and os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
            try:
                embeddings = GoogleGenerativeAIEmbeddings(
                    model=EMBEDDING_MODEL,
                    google_api_key=GOOGLE_API_KEY
                )
                self.vectorstore = FAISS.load_local(
                    VECTOR_STORE_PATH, 
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                self.log("Vector database loaded successfully!")
                self.update_files_list()
                self.create_qa_chain()
            except Exception as e:
                self.log(f"Failed to load vector database: {e}")
    
    def update_files_list(self):
        self.files_listbox.delete(0, tk.END)
        if os.path.exists("embedded_files.txt"):
            with open("embedded_files.txt", "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.files_listbox.insert(tk.END, line.strip())
                        self.embedded_files.append(line.strip())
    
    def create_qa_chain(self):
        if not self.vectorstore:
            return
        try:
            llm = ChatGoogleGenerativeAI(
                model=LLM_MODEL,
                google_api_key=GOOGLE_API_KEY,
                temperature=0.7,
                convert_system_message_to_human=True
            )
            
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": TOP_K}
            )
            
            prompt_template = """You are a professional knowledge Q&A assistant. Please answer based on the following context.

Context:
{context}

Question: {question}

Please provide a clear and accurate answer."""

            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
            self.log("QA chain created!")
            self.status_var.set("Ready")
        except Exception as e:
            self.log(f"Failed to create QA chain: {e}")
    
    def add_files(self):
        file_types = [
            ("All Supported", "*.pdf *.txt *.docx *.doc *.xlsx *.xls *.epub"),
            ("PDF", "*.pdf"),
            ("Text", "*.txt"),
            ("Word", "*.docx *.doc"),
            ("Excel", "*.xlsx *.xls"),
            ("EPUB", "*.epub")
        ]
        
        files = filedialog.askopenfilenames(title="Select files to embed", filetypes=file_types)
        
        if files:
            self.embed_files(files)
    
    def embed_files(self, file_paths):
        self.stop_embedding = False
        self.status_var.set("Embedding...")
        
        def embed_thread():
            try:
                from pypdf import PdfReader
                
                documents = []
                extensions = {".pdf": "PDF", ".txt": "TXT", ".docx": "DOCX", ".doc": "DOCX",
                            ".xlsx": "XLSX", ".xls": "XLSX", ".epub": "EPUB"}
                
                for file_path in file_paths:
                    if self.stop_embedding:
                        break
                    
                    file_path = Path(file_path)
                    ext = file_path.suffix.lower()
                    
                    self.log(f"Loading: {file_path.name}")
                    
                    try:
                        if ext == ".pdf":
                            reader = PdfReader(str(file_path))
                            for i, page in enumerate(reader.pages):
                                text = page.extract_text()
                                if text and text.strip():
                                    doc = Document(
                                        page_content=text,
                                        metadata={"source": file_path.name, "page": i+1}
                                    )
                                    documents.append(doc)
                            
                        elif ext == ".txt":
                            loader = TextLoader(str(file_path), encoding="utf-8")
                            documents.extend(loader.load())
                            
                        elif ext in [".docx", ".doc"]:
                            loader = Docx2txtLoader(str(file_path))
                            documents.extend(loader.load())
                            
                        elif ext in [".xlsx", ".xls"]:
                            loader = UnstructuredExcelLoader(str(file_path))
                            documents.extend(loader.load())
                            
                        elif ext == ".epub":
                            loader = UnstructuredEPubLoader(str(file_path))
                            documents.extend(loader.load())
                        
                        self.log(f"   Loaded: {file_path.name}")
                        
                    except Exception as e:
                        self.log(f"   Failed: {file_path.name} - {e}")
                
                if not documents:
                    self.log("No documents to embed")
                    return
                
                self.log(f"Total documents: {len(documents)}")
                
                # Split documents
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    length_function=len,
                    add_start_index=True
                )
                
                splits = text_splitter.split_documents(documents)
                self.log(f"Split into {len(splits)} chunks")
                
                # Create or update vectorstore
                if self.vectorstore:
                    self.log("Adding to existing vector database...")
                    from langchain_google_genai import GoogleGenerativeAIEmbeddings
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model=EMBEDDING_MODEL,
                        google_api_key=GOOGLE_API_KEY
                    )
                    self.vectorstore.add_documents(splits)
                else:
                    self.log("Creating new vector database...")
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model=EMBEDDING_MODEL,
                        google_api_key=GOOGLE_API_KEY
                    )
                    self.vectorstore = FAISS.from_documents(
                        documents=splits,
                        embedding=embeddings
                    )
                
                # Save vectorstore
                self.vectorstore.save_local(VECTOR_STORE_PATH)
                self.log("Vector database saved!")
                
                # Save embedded files list
                with open("embedded_files.txt", "w", encoding="utf-8") as f:
                    for fp in file_paths:
                        f.write(Path(fp).name + "\n")
                
                self.root.after(0, self.update_files_list)
                self.root.after(0, lambda: self.create_qa_chain() or self.status_var.set("Ready"))
                self.log("Embedding complete!")
                
            except Exception as e:
                self.log(f"Embedding error: {e}")
                self.status_var.set("Error")
        
        self.embedding_thread = threading.Thread(target=embed_thread, daemon=True)
        self.embedding_thread.start()
    
    def reembed_all(self):
        files = filedialog.askopenfilenames(title="Select files to re-embed", 
                                            filetypes=[("All", "*.pdf *.txt *.docx *.xlsx *.epub")])
        if files:
            # Clear existing
            if os.path.exists(VECTOR_STORE_PATH):
                import shutil
                shutil.rmtree(VECTOR_STORE_PATH)
            open("embedded_files.txt", "w").close()
            self.vectorstore = None
            self.embedded_files = []
            self.files_listbox.delete(0, tk.END)
            self.embed_files(files)
    
    def clear_database(self):
        if messagebox.askyesno("Confirm", "Clear all embedded files and database?"):
            if os.path.exists(VECTOR_STORE_PATH):
                import shutil
                shutil.rmtree(VECTOR_STORE_PATH)
            open("embedded_files.txt", "w").close()
            self.vectorstore = None
            self.qa_chain = None
            self.embedded_files = []
            self.files_listbox.delete(0, tk.END)
            self.log("Database cleared!")
    
    def ask_question(self, event=None):
        question = self.question_entry.get().strip()
        if not question:
            return
        
        if not self.qa_chain:
            messagebox.showwarning("Warning", "Please embed some documents first!")
            return
        
        self.question_entry.delete(0, tk.END)
        self.status_var.set("Thinking...")
        self.answer_text.config(state=tk.NORMAL)
        self.answer_text.delete(1.0, tk.END)
        self.answer_text.insert(tk.END, "Thinking...\n")
        self.answer_text.config(state=tk.DISABLED)
        self.root.update()
        
        def ask_thread():
            try:
                result = self.qa_chain({"query": question})
                
                self.root.after(0, lambda: self.answer_text.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.answer_text.delete(1.0, tk.END))
                self.root.after(0, lambda: self.answer_text.insert(tk.END, result['result']))
                self.root.after(0, lambda: self.answer_text.config(state=tk.DISABLED))
                self.root.after(0, lambda: self.status_var.set("Ready"))
                
            except Exception as e:
                self.root.after(0, lambda: self.answer_text.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.answer_text.delete(1.0, tk.END))
                self.root.after(0, lambda: self.answer_text.insert(tk.END, f"Error: {e}"))
                self.root.after(0, lambda: self.answer_text.config(state=tk.DISABLED))
                self.root.after(0, lambda: self.status_var.set("Error"))
        
        threading.Thread(target=ask_thread, daemon=True).start()


def main():
    root = tk.Tk()
    app = RAGApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()