# src/qa_system.py

# Import the function to load the index
from src.rag_pipeline import load_faiss_index
# Core LlamaIndex imports
from llama_index.core import Settings, PromptTemplate
# Embedding and LLM imports
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
# Standard library
import logging
import torch
# Optional: Keep import if you might try quantization later
# from transformers import BitsAndBytesConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().setLevel(logging.INFO) # Ensure root logger level

# --- Configure Global Settings for LlamaIndex ---
logging.info("Configuring global settings for LlamaIndex...")

# 1. Configure Embedding Model (Keep as is)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
logging.info(f"Using embedding model: {Settings.embed_model.model_name}")

# 2. Configure LLM to use Llama 3.2 3B Instruct <<<--- CHANGE HERE
llm_model_name = "meta-llama/Llama-3.2-1B-Instruct" # <<<--- NEW MODEL
logging.info(f"Setting up LLM: {llm_model_name}")

# --- Define a RAG-specific prompt template (Llama 3 template should work) ---
query_wrapper_prompt = PromptTemplate(
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are an expert Q&A assistant specialized in neural network architectures. "
    "Your goal is to answer the user's query accurately based *only* on the provided context information. "
    "If the context does not contain the information needed to answer the query, "
    "state that the answer is not found in the context. Do not add information "
    "that is not present in the context. Keep your answers concise and directly relevant to the query."
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "Context information:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the query.\n"
    "Query: {query_str}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)
# ---------------------------------------------------------------------

# --- LLM Initialization (Quantization DISABLED by default for 3B model) ---
logging.info("Initializing LLM (Quantization Disabled).")
Settings.llm = HuggingFaceLLM(
    model_name=llm_model_name,
    tokenizer_name=llm_model_name,
    query_wrapper_prompt=query_wrapper_prompt,
    context_window=131072, # Llama 3.2 3B also supports 128k context
    max_new_tokens=512, # Keep reasonably high, adjust if needed
    model_kwargs={
        # Removed quantization_config
        # Removed torch_dtype=torch.float16 - let transformers decide best default
        # (often float32 on CPU/low VRAM, float16 on sufficient VRAM)
    },
    generate_kwargs={
        "temperature": 0.7,
        "do_sample": True,
    },
    device_map="auto", # Automatically use GPU if possible
)
logging.info(f"LLM '{llm_model_name}' configured successfully (No Quantization).")
# -------------------------------------------------

def answer_nn_question(question: str) -> str:
    """Loads the index, creates a query engine, and answers a question using the configured LLM."""
    logging.info(f"Received question: '{question}'")

    logging.info("Loading index...")
    index = load_faiss_index(persist_dir="storage")
    if index is None:
        logging.error("Index loading failed.")
        return "Error: Could not load the index. Please build it first using rag_pipeline.py."
    logging.info("Index loaded successfully.")

    logging.info("Creating query engine...")
    query_engine = index.as_query_engine()
    logging.info("Query engine ready.")

    logging.info(f"Querying with: '{question}'")
    try:
        logging.info("Sending query to LLM...")
        response = query_engine.query(question)
        logging.info("LLM processing finished. Received response object.")
        logging.debug(f"Raw response type: {type(response)}")
        logging.debug(f"Raw response content: {response}")

        answer_text = str(response.response).strip()
        # Llama 3.2 likely uses <|eot_id|> too
        if answer_text.endswith("<|eot_id|>"):
              answer_text = answer_text[:-len("<|eot_id|>")].strip()
        logging.info("Successfully processed LLM response.")
        return answer_text
    except Exception as e:
        logging.error(f"An error occurred during querying: {e}", exc_info=True)
        return f"Error during query: {e}"

# --- Main execution block ---
if __name__ == "__main__":
    logging.info("Starting QA System with Llama 3.2 1B LLM...")
    user_question = input("Ask a question about neural networks (e.g., 'What is ResNet?'): ")
    if user_question:
        answer = answer_nn_question(user_question)
        print("\nSynthesized Answer (Llama 3.2 1B):") # Updated model name here
        print(answer)
    else:
        print("No question asked.")

