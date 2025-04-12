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
# TransformComponent is no longer needed here
from llama_index.core.schema import BaseNode, Document # Added Document import
from typing import List # For type hinting
# ---
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.settings import Settings # Import Settings for type hint/context
# FAISS integration
from llama_index.vector_stores.faiss import FaissVectorStore
# HuggingFace Embedding integration
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# FAISS library
import faiss
# Standard libraries
import os
import logging
import json # For loading metadata

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)
logging.getLogger(__name__).setLevel(logging.DEBUG) # Enable debug for this file

# --- Default Model Names (can be overridden) ---
DEFAULT_EMBED_MODEL = "BAAI/bge-base-en-v1.5" # Using BGE-Base now

# --- Text Cleaning Function ---
def clean_text_for_utf8(text: str) -> str:
    """Removes or replaces characters that cause UTF-8 encoding errors, like lone surrogates."""
    # Using errors='replace' inserts '?' for invalid characters
    return text.encode('utf-8', 'replace').decode('utf-8')

# --- Helper Function to Load Docs with Metadata (NOW INCLUDES CLEANING) ---
def load_documents_with_metadata(data_dir="data/research_papers"):
    """Loads PDFs, attaches metadata, and cleans text content."""
    pdf_dir = os.path.join(data_dir, "raw_pdfs")
    metadata_file = os.path.join(data_dir, "arxiv_metadata.json")

    # --- Metadata Loading (same as before) ---
    if not os.path.isdir(pdf_dir):
        logging.error(f"PDF directory not found: {pdf_dir}")
        return []
    if not os.path.isfile(metadata_file):
        logging.warning(f"Metadata JSON file not found: {metadata_file}. Loading PDFs without detailed metadata.")
        metadata_lookup = {} # Empty lookup
    else:
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                all_metadata = json.load(f)
            metadata_lookup = {meta.get("local_pdf_path"): meta for meta in all_metadata if meta.get("local_pdf_path")}
        except Exception as e:
            logging.error(f"Failed to load or parse metadata file {metadata_file}: {e}. Proceeding without detailed metadata.")
            metadata_lookup = {}
    # -------------------------------------------

    # --- File Metadata Function (same as before) ---
    def file_metadata_func(file_path: str):
        meta = metadata_lookup.get(file_path)
        if meta:
            return {
                "file_path": file_path, "file_name": os.path.basename(file_path),
                "title": meta.get("title", "N/A"), "authors": ", ".join(meta.get("authors", [])),
                "published": meta.get("published", "N/A"), "arxiv_id": meta.get("arxiv_id", "N/A"),
                "pdf_url": meta.get("pdf_url", "N/A"),
            }
        else:
             return {"file_path": file_path, "file_name": os.path.basename(file_path), "title": "N/A (Metadata not found)"}
    # -------------------------------------------

    # --- Load Documents ---
    reader = SimpleDirectoryReader(pdf_dir, file_metadata=file_metadata_func)
    try:
        documents = reader.load_data()
        logging.info(f"Loaded {len(documents)} documents initialy.")
    except Exception as e:
        logging.error(f"Failed during SimpleDirectoryReader.load_data(): {e}", exc_info=True)
        return []
    # ----------------------

    # --- <<< NEW: Clean Document Content >>> ---
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
             # Create a new Document or modify in place if safe
             # Modifying might be okay here since it's before pipeline
             doc.set_content(cleaned_content)
             cleaned_documents.append(doc)
         except Exception as e:
             logging.error(f"Error cleaning doc {i} (file: {doc.metadata.get('file_name', 'N/A')}): {e}", exc_info=False)
             # Optionally skip the document: continue
             cleaned_documents.append(doc) # Or keep original if cleaning failed? Decide policy. Let's keep it for now.

    logging.info(f"Finished cleaning. Content modified in {docs_cleaned_count} document(s). Returning {len(cleaned_documents)} documents.")
    # ------------------------------------------
    return cleaned_documents # Return the list of cleaned documents


# --- REMOVED TextCleanerComponent class ---

# --- Enhanced Index Building Function ---
def build_faiss_index(
    window_size=3
    # No embed_model_name parameter here anymore
):
    # Define persist_dir internally for use within the function
    persist_dir = "storage"
    # Use the constant defined at the top of the file for logging
    logging.info(f"Starting FAISS index build. Embed Model: {DEFAULT_EMBED_MODEL}") # <-- FIXED
    logging.info(f"Index will be persisted to: '{persist_dir}'") # Optional: Log the path
    # 1. Initialize Embedding Model
    try:
        embed_model = HuggingFaceEmbedding(model_name=DEFAULT_EMBED_MODEL) # Initialize using the constant
        embed_dim = 768 # Dimension for BAAI/bge-base-en-v1.5
        # Now you could log embed_model.model_name if needed, but using the constant is fine here.
        logging.info(f"Embedding model '{DEFAULT_EMBED_MODEL}' initialized successfully. Dimension: {embed_dim}")
    except Exception as e:
        logging.error(f"Failed to load embedding model '{DEFAULT_EMBED_MODEL}': {e}", exc_info=True)
        return

    # 2. Define Node Parser (Sentence Window)
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    # 3. Load AND CLEAN Documents with Metadata
    # Call the function without the undefined 'data_dir' variable
    # It will use its own internal default ("data/research_papers")
    documents = load_documents_with_metadata() # <-- FIXED: Argument removed
    if not documents:
        logging.error("No documents loaded or available after cleaning. Aborting index build.")
        return

    # 4. Create and Run Ingestion Pipeline (Cleaner component REMOVED)
    logging.info("Defining ingestion pipeline...")
    pipeline = IngestionPipeline(
        transformations=[
            # Cleaner component removed - cleaning happened during load
            node_parser,
            embed_model,
        ]
    )
    # 5. Prepare FAISS Index
    logging.info("Running ingestion pipeline (parsing and embedding)... This may take time.")
    try:
        # Now pass the CLEANED documents to the pipeline
        nodes = pipeline.run(documents=documents, show_progress=True)
    except Exception as e:
        # Catch potential errors during the pipeline run
        logging.error(f"Error during ingestion pipeline run: {e}", exc_info=True)
        return

    logging.info(f"Ingestion pipeline created {len(nodes)} nodes.")
    if not nodes:
        logging.error("Pipeline did not produce any nodes. Aborting.")
        return

    # 6. Prepare FAISS Index
    logging.info(f"Initializing FAISS index (IndexFlatL2) with dimension: {embed_dim}")
    faiss_index = faiss.IndexFlatL2(embed_dim)

    # 7. Create Vector Store and Storage Context
    logging.info("Creating FAISS Vector Store...")
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 8. Add Nodes to Index
    logging.info("Adding embedded nodes to the vector store...")
    try:
        vector_store.add(nodes) # Add nodes that already have embeddings from the pipeline
        logging.info("Nodes added to vector store.")
    except Exception as e:
        logging.error(f"Error adding nodes to vector store: {e}", exc_info=True)
        return

    # 9. Create the LlamaIndex VectorStoreIndex object
    index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, embed_model=embed_model)

    # 10. Persist Index to Disk
    faiss_binary_path = os.path.join(persist_dir, "vector_store.faiss")
    logging.info(f"Persisting index metadata and FAISS binary to '{persist_dir}'")
    os.makedirs(persist_dir, exist_ok=True)
    try:
        index.storage_context.persist(persist_dir=persist_dir)
        logging.info(f"Saving FAISS binary index to: {faiss_binary_path}")
        faiss.write_index(faiss_index, faiss_binary_path)
        logging.info("Index built and persisted successfully.")
    except Exception as e:
        logging.error(f"Failed to persist index or FAISS binary: {e}", exc_info=True)

# --- Enhanced Index Loading Function (No changes needed from previous version) ---
def load_faiss_index(
    persist_dir="storage",
    embed_model_name=DEFAULT_EMBED_MODEL # Need to know which model was used
):
    """
    Loads the FAISS index by explicitly loading the binary file
    and ensuring the correct embedding model is configured in Settings.
    """
    logging.info(f"Attempting to load index from '{persist_dir}' (Embed Model: {embed_model_name})")
    faiss_binary_path = os.path.join(persist_dir, "vector_store.faiss")

    # Crucially, configure the *correct* embedding model globally before loading
    try:
        logging.info(f"Configuring Settings.embed_model to '{embed_model_name}' for loading.")
        Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
    except Exception as e:
        logging.error(f"Failed to load embedding model '{embed_model_name}' during index load setup: {e}", exc_info=True)
        return None

    if not os.path.exists(persist_dir) or not os.path.exists(faiss_binary_path):
        logging.error(f"Index directory '{persist_dir}' or FAISS binary '{faiss_binary_path}' not found.")
        return None

    try:
        logging.info(f"Loading FAISS binary index from: {faiss_binary_path}")
        faiss_index = faiss.read_index(faiss_binary_path)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        logging.info("Loading LlamaIndex storage context (metadata)...")
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=persist_dir
        )
        logging.info("Loading main VectorStoreIndex object...")
        index = load_index_from_storage(storage_context)
        logging.info("FAISS index and LlamaIndex context loaded successfully.")
        return index
    except Exception as e:
        logging.error(f"Failed to load index from '{persist_dir}': {e}", exc_info=True)
        return None

# --- Main execution block for direct testing of index build ---
if __name__ == '__main__':
    logging.info("Executing rag_pipeline.py script directly (for index building).")
    # Example: Build the index directly if the script is run
    # Remove all unexpected keyword arguments from the CALL
    build_faiss_index(
        # embed_model_name=DEFAULT_EMBED_MODEL # <--- REMOVE THIS LINE
        # window_size=3 # Can still pass other valid args like window_size
    )
    print("\nDirect build process complete.")