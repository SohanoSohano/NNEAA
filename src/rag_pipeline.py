# src/rag_pipeline.py

# Core LlamaIndex imports
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
    load_index_from_storage # Make sure this is imported
)
# FAISS integration
from llama_index.vector_stores.faiss import FaissVectorStore
# HuggingFace Embedding integration
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- Imports needed for loading ---
from llama_index.core import load_index_from_storage

# FAISS library
import faiss
# Standard libraries
import os
import logging

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configure Global Settings for LlamaIndex ---
# Set the embedding model globally *before* using it in index creation or loading.
# This ensures LlamaIndex uses this model instead of defaulting to OpenAI.
logging.info("Configuring global settings for LlamaIndex...")
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    # Optional: specify device, e.g., 'cuda' if you have a GPU, 'cpu' otherwise
    # device='cuda'
)
# We are not using an LLM in this specific script (only embeddings),
# so we can set it to None or leave it as default (which might still try OpenAI if not careful).
# Setting it explicitly avoids potential issues.
Settings.llm = None
logging.info(f"Using embedding model: {Settings.embed_model.model_name}")
# -------------------------------------------------

def build_faiss_index(data_dir="data/research_papers", persist_dir="storage"):
    """
    Builds a FAISS index using the globally configured embedding model.
    Args:
        data_dir (str): Directory containing documents to index.
        persist_dir (str): Directory to save the built index.
    """
    logging.info(f"Starting FAISS index build process. Data directory: {data_dir}")

    # 1. Load Documents
    try:
        logging.info(f"Loading documents from '{data_dir}'...")
        documents = SimpleDirectoryReader(data_dir).load_data()
        if not documents:
            logging.warning(f"No documents found in '{data_dir}'. Index will be empty.")
            # Optionally create an empty index or return early
            # For now, we let it proceed, FAISS can handle empty data.
        else:
            logging.info(f"Successfully loaded {len(documents)} documents.")
    except Exception as e:
        logging.error(f"Failed to load documents from '{data_dir}': {e}", exc_info=True)
        return # Stop execution if documents can't be loaded

    # 2. Prepare FAISS Index
    # Dimension must match the embedding model's output dimension.
    # 'all-MiniLM-L6-v2' has 384 dimensions.
    # Dimension must match the embedding model's output dimension.
    # 'all-MiniLM-L6-v2' has 384 dimensions.
    embed_dim = 384 # <-- FIXED LINE

    logging.info(f"Initializing FAISS index with dimension: {embed_dim}")
    faiss_index = faiss.IndexFlatL2(embed_dim)

    # 3. Create Vector Store and Storage Context
    logging.info("Creating FAISS Vector Store...")
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 4. Index Documents
    # LlamaIndex will automatically use the embed_model set in Settings
    logging.info("Indexing documents... This may take some time depending on document count and size.")
    try:
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True # Display a progress bar
        )
    except Exception as e:
        logging.error(f"Failed during document indexing: {e}", exc_info=True)
        return

    # 5. Persist LlamaIndex Metadata AND FAISS Binary Index
    logging.info(f"Persisting index metadata and FAISS binary to directory: '{persist_dir}'")
    os.makedirs(persist_dir, exist_ok=True)
    faiss_binary_path = os.path.join(persist_dir, "vector_store.faiss") # Define path for FAISS binary

    try:
        # Persist LlamaIndex metadata (docstore.json, index_store.json, etc.)
        index.storage_context.persist(persist_dir=persist_dir)

        # --- Explicitly save the FAISS index as a binary file ---
        logging.info(f"Saving FAISS binary index to: {faiss_binary_path}")
        faiss.write_index(faiss_index, faiss_binary_path)
        # ---------------------------------------------------------

        logging.info("FAISS index built and persisted successfully (metadata + binary).")
    except Exception as e:
        logging.error(f"Failed to persist index or FAISS binary: {e}", exc_info=True)


def load_faiss_index(persist_dir="storage"):
    """
    Loads the FAISS index by explicitly loading the binary file
    and then constructing the StorageContext.
    Args:
        persist_dir (str): Directory containing the saved index metadata and binary.
    Returns:
        VectorStoreIndex or None: The loaded index, or None if loading fails.
    """
    logging.info(f"Attempting to load index from directory: '{persist_dir}'")
    faiss_binary_path = os.path.join(persist_dir, "vector_store.faiss")

    # Ensure Settings.embed_model is set before loading
    if not hasattr(Settings, 'embed_model') or Settings.embed_model is None:
         logging.warning("Settings.embed_model not configured. Configuring now.")
         Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
         )
         Settings.llm = None

    if not os.path.exists(persist_dir) or not os.path.exists(faiss_binary_path):
        logging.error(f"Index directory '{persist_dir}' or FAISS binary '{faiss_binary_path}' not found. Cannot load index.")
        return None

    try:
        # --- 1. Explicitly load the FAISS binary index ---
        logging.info(f"Loading FAISS binary index from: {faiss_binary_path}")
        faiss_index = faiss.read_index(faiss_binary_path)
        # --------------------------------------------------

        # --- 2. Create the FaissVectorStore from the loaded binary index ---
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        # -------------------------------------------------------------------

        # --- 3. Create StorageContext, providing the loaded vector_store AND persist_dir ---
        # persist_dir is still needed for LlamaIndex to find its own metadata files (docstore.json etc.)
        logging.info("Loading LlamaIndex storage context (metadata)...")
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, # Pass the loaded vector store
            persist_dir=persist_dir      # Tell it where to find other metadata
        )
        # ------------------------------------------------------------------------------------

        # --- 4. Load the main LlamaIndex object ---
        logging.info("Loading main VectorStoreIndex...")
        # LlamaIndex uses the embed_model from Settings automatically
        index = load_index_from_storage(storage_context)
        # ------------------------------------------

        logging.info("FAISS index and LlamaIndex context loaded successfully.")
        return index
    except Exception as e:
        logging.error(f"Failed to load index from '{persist_dir}': {e}", exc_info=True)
        return None

# --- Main execution block for direct testing ---
if __name__ == '__main__':
    logging.info("Executing rag_pipeline.py script directly.")
    # Example: Build the index if the script is run directly
    build_faiss_index(data_dir="data/research_papers", persist_dir="storage")

    # Example: Test loading the index immediately after building
    logging.info("Attempting to load the index just built...")
    loaded_index = load_faiss_index(persist_dir="storage")
    if loaded_index:
        logging.info("Index loaded successfully after build. Ready for querying (in another script).")
    else:
        logging.error("Failed to load index after building.")

