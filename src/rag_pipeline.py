# src/rag_pipeline.py

# Core LlamaIndex imports
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage
)
# Correct Imports for Node Parser and Transformation Base Class
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.schema import BaseNode, Document  # Added Document import
from typing import List  # For type hinting

from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.settings import Settings  # Import Settings for type hint/context

# FAISS integration
from llama_index.vector_stores.faiss import FaissVectorStore

# HuggingFace Embedding integration
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# FAISS library
import faiss

# Standard libraries
import os
import logging
import json  # For loading metadata
import shutil  # For directory operations

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)
logging.getLogger(__name__).setLevel(logging.DEBUG)  # Enable debug for this file

# --- Default Model Names (can be overridden) ---
DEFAULT_EMBED_MODEL = "BAAI/bge-base-en-v1.5"  # Using BGE-Base now

# --- Text Cleaning Function ---
def clean_text_for_utf8(text: str) -> str:
    """Removes or replaces characters that cause UTF-8 encoding errors."""
    if not isinstance(text, str):
        return ""
    # More aggressive cleaning - only keep ASCII characters
    return ''.join(char for char in text if ord(char) < 128)

# --- Helper Function to Load Docs with Metadata ---
def load_documents_with_metadata(data_dir="data/research_papers"):
    """Loads PDFs, attaches metadata, and cleans text content."""
    pdf_dir = os.path.join(data_dir, "raw_pdfs")
    metadata_file = os.path.join(data_dir, "arxiv_metadata.json")

    if not os.path.isdir(pdf_dir):
        logging.error(f"PDF directory not found: {pdf_dir}")
        return []

    if not os.path.isfile(metadata_file):
        logging.warning(f"Metadata JSON file not found: {metadata_file}. Loading PDFs without detailed metadata.")
        metadata_lookup = {}  # Empty lookup
    else:
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                all_metadata = json.load(f)
                metadata_lookup = {meta.get("local_pdf_path"): meta for meta in all_metadata if meta.get("local_pdf_path")}
        except Exception as e:
            logging.error(f"Failed to load or parse metadata file {metadata_file}: {e}. Proceeding without detailed metadata.")
            metadata_lookup = {}

    # --- File Metadata Function ---
    def file_metadata_func(file_path: str):
        """Creates metadata dict and cleans string values for UTF-8 compatibility."""
        # Basic metadata
        base_meta = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
        }
        # Look up additional metadata
        arxiv_meta_raw = metadata_lookup.get(file_path)
        if arxiv_meta_raw:
            # Add fields from arxiv_metadata.json, CLEANING string values
            base_meta["title"] = clean_text_for_utf8(arxiv_meta_raw.get("title", "N/A"))
            base_meta["authors"] = clean_text_for_utf8(", ".join(arxiv_meta_raw.get("authors", [])))
            base_meta["published"] = clean_text_for_utf8(arxiv_meta_raw.get("published", "N/A"))
            base_meta["arxiv_id"] = clean_text_for_utf8(arxiv_meta_raw.get("arxiv_id", "N/A"))
            base_meta["pdf_url"] = clean_text_for_utf8(arxiv_meta_raw.get("pdf_url", "N/A"))
        else:
            base_meta["title"] = "N/A (Metadata not found)"
        return base_meta

    # --- Load Documents ---
    try:
        reader = SimpleDirectoryReader(pdf_dir, file_metadata=file_metadata_func)
        documents = reader.load_data()
        logging.info(f"Loaded {len(documents)} documents with metadata attached.")
    except Exception as e:
        logging.error(f"Failed during SimpleDirectoryReader.load_data(): {e}", exc_info=True)
        return []

    # --- Clean Document CONTENT ---
    cleaned_documents = []
    logging.info("Cleaning document content for UTF-8 compatibility...")
    docs_cleaned_count = 0
    for i, doc in enumerate(documents):
        try:
            original_content = doc.get_content()
            cleaned_content = clean_text_for_utf8(original_content)
            if cleaned_content != original_content:
                docs_cleaned_count += 1
                logging.debug(f"Cleaned content for doc {i} (file: {doc.metadata.get('file_name', 'N/A')})")
            doc.set_content(cleaned_content)
            cleaned_documents.append(doc)
        except Exception as e:
            logging.error(f"Error cleaning doc {i} (file: {doc.metadata.get('file_name', 'N/A')}): {e}", exc_info=False)
            # Try to create a new document with minimal content
            try:
                minimal_content = "Document content could not be processed due to encoding issues."
                new_doc = Document(text=minimal_content, metadata=doc.metadata)
                cleaned_documents.append(new_doc)
                logging.warning(f"Created replacement document for {doc.metadata.get('file_name', 'N/A')}")
            except:
                logging.error(f"Could not create replacement document for {i}")

    logging.info(f"Finished content cleaning. Content modified in {docs_cleaned_count} document(s). Returning {len(cleaned_documents)} documents.")
    return cleaned_documents

# --- Check Storage Integrity Function ---
def check_storage_integrity(persist_dir):
    """Checks if storage directory has all required files."""
    required_files = ['docstore.json', 'index_store.json', 'vector_store.json', 'vector_store.faiss']
    missing_files = []
    
    for file in required_files:
        path = os.path.join(persist_dir, file)
        if not os.path.exists(path):
            missing_files.append(file)
            logging.warning(f"Missing file: {path}")
        else:
            logging.info(f"File exists: {path}")
    
    return len(missing_files) == 0

# --- Revised Index Building Function ---
def build_faiss_index(window_size=3):
    persist_dir = "storage"
    logging.info(f"Starting FAISS index build. Embed Model: {DEFAULT_EMBED_MODEL}")
    logging.info(f"Index will be persisted to: '{persist_dir}'")

    # If storage directory exists, back it up and create fresh
    if os.path.exists(persist_dir):
        backup_dir = f"{persist_dir}_backup_{int(time.time())}"
        logging.info(f"Backing up existing storage directory to {backup_dir}")
        try:
            shutil.copytree(persist_dir, backup_dir)
            shutil.rmtree(persist_dir)
            logging.info("Backup completed and old storage directory removed")
        except Exception as e:
            logging.error(f"Error during backup: {e}")
            return

    # 1. Initialize Embedding Model
    try:
        # Set embed_model globally via Settings for consistency
        Settings.embed_model = HuggingFaceEmbedding(model_name=DEFAULT_EMBED_MODEL)
        embed_dim = 768
        logging.info(f"Embedding model '{DEFAULT_EMBED_MODEL}' set globally. Dimension: {embed_dim}")
    except Exception as e:
        logging.error(f"Failed to load embedding model '{DEFAULT_EMBED_MODEL}': {e}", exc_info=True)
        return

    # 2. Define Node Parser
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    # 3. Load AND CLEAN Documents
    documents = load_documents_with_metadata()
    if not documents:
        logging.error("No documents loaded. Aborting.")
        return

    # --- Simplified Index Creation & Persistence ---
    try:
        # 4. Create FAISS Index and Vector Store
        logging.info(f"Initializing FAISS index (IndexFlatL2) with dimension: {embed_dim}")
        faiss_index = faiss.IndexFlatL2(embed_dim)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        
        # 5. Create Storage Context (associating the vector store)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # 6. Build VectorStoreIndex FROM documents
        logging.info("Building VectorStoreIndex from documents (parsing, embedding, indexing)...")
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            transformations=[node_parser],  # Apply parsing during indexing
            show_progress=True,
        )
        
        logging.info(f"Index built with {len(index.docstore.docs)} nodes in docstore.")
        
        # 7. Persist index
        os.makedirs(persist_dir, exist_ok=True)
        logging.info(f"Persisting index metadata to '{persist_dir}'")
        index.storage_context.persist(persist_dir=persist_dir)
        
        # 8. Save FAISS binary separately
        faiss_binary_path = os.path.join(persist_dir, "vector_store.faiss")
        logging.info(f"Saving FAISS binary index to: {faiss_binary_path}")
        faiss.write_index(faiss_index, faiss_binary_path)
        
        # 9. Verify storage integrity
        if check_storage_integrity(persist_dir):
            logging.info("Index built and persisted successfully with all required files.")
        else:
            logging.warning("Index built but some files may be missing. Check the logs.")
            
    except Exception as e:
        logging.error(f"Error during index building/persistence: {e}", exc_info=True)

# --- Revised Index Loading Function ---
def load_faiss_index(persist_dir="storage", embed_model_name=DEFAULT_EMBED_MODEL):
    logging.info(f"Attempting to load index from '{persist_dir}' (Embed Model: {embed_model_name})")
    
    # Check storage integrity first
    if not check_storage_integrity(persist_dir):
        logging.error("Storage integrity check failed. Some required files are missing.")
        return None
    
    # Configure embedding model
    try:
        logging.info(f"Configuring Settings.embed_model to '{embed_model_name}' for loading.")
        Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
    except Exception as e:
        logging.error(f"Failed to load embedding model '{embed_model_name}': {e}", exc_info=True)
        return None

    # Load FAISS binary directly first to verify it works
    faiss_binary_path = os.path.join(persist_dir, "vector_store.faiss")
    try:
        logging.info(f"Loading FAISS binary from {faiss_binary_path}")
        faiss_index = faiss.read_index(faiss_binary_path)
        logging.info(f"FAISS binary loaded successfully with {faiss_index.ntotal} vectors")
    except Exception as e:
        logging.error(f"Failed to load FAISS binary: {e}", exc_info=True)
        return None

    # Try loading the storage context with different encodings
    try:
        logging.info("Loading LlamaIndex storage context (metadata)...")
        
        # Simplified approach - create a fresh vector store from the binary
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        
        # Create storage context with the loaded vector store
        storage_context = StorageContext.from_defaults(
            persist_dir=persist_dir,
            vector_store=vector_store
        )
        
        logging.info("Successfully loaded storage context")
        
        # Load the index from storage
        logging.info("Loading main VectorStoreIndex object...")
        index = load_index_from_storage(storage_context)
        logging.info("FAISS index and LlamaIndex context loaded successfully.")
        return index
        
    except Exception as e:
        logging.error(f"Failed to load index from '{persist_dir}': {e}", exc_info=True)
        return None

# --- Main execution block ---
if __name__ == '__main__':
    import time  # For timestamping backups
    
    logging.info("Executing rag_pipeline.py script directly (for index building).")
    build_faiss_index()  # Call simplified build function
    print("\nDirect build process complete.")
