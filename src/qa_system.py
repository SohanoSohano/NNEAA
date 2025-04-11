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
from transformers import BitsAndBytesConfig
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configure Global Settings for LlamaIndex ---
logging.info("Configuring global settings for LlamaIndex...")

# 1. Configure Embedding Model (Keep as is)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
logging.info(f"Using embedding model: {Settings.embed_model.model_name}")

# 2. Configure LLM to use Llama 3.1 8B Instruct <<< CHANGE HERE
# llm_model_name = "meta-llama/Meta-Llama-3-8B-Instruct" # Old Llama 3
llm_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct" # <<< NEW: Llama 3.1
logging.info(f"Setting up LLM: {llm_model_name}")

# --- Define a RAG-specific prompt template (Keep the Llama 3 template, it's usually compatible) ---
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

# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_compute_dtype=torch.float16 # Optional: Recommended for faster computation if GPU supports it
   # You might need other bnb_* args depending on your bitsandbytes version/setup
)
logging.info("Using 4-bit quantization configuration.")

Settings.llm = HuggingFaceLLM(
    model_name=llm_model_name,
    tokenizer_name=llm_model_name, # <<< Use the same name for tokenizer
    query_wrapper_prompt=query_wrapper_prompt,
    # context_window=8192, # Old Llama 3 context
    context_window=131072, # <<< NEW: Llama 3.1 supports 128k context (131072 = 128 * 1024)
    max_new_tokens=50,
    model_kwargs={
        # --- Quantization for Memory Saving ---
        #"quantization_config": quantization_config
        #"load_in_4bit": True,
        # Comment out above and uncomment below if you prefer float16 and have enough VRAM (>16GB)
        "torch_dtype": torch.float16
    },
    generate_kwargs={
        "temperature": 0.7,
        "do_sample": True,
    },
    device_map="auto",
)
logging.info("LLM configured successfully. Using Llama 3.1 8B Instruct.")
# -------------------------------------------------

# ... (Keep the rest of the file: answer_nn_question function, if __name__ == "__main__": block) ...

# Make sure the answer cleaning in answer_nn_question still works:
def answer_nn_question(question: str) -> str:
    """Loads the index, creates a query engine, and answers a question using the configured LLM."""

    # Log the start of the function call
    logging.info(f"Received question: '{question}'")
    # No need to re-configure Settings here, rely on global config.

    # 1. Load the pre-built index
    logging.info("Loading index...")
    index = load_faiss_index(persist_dir="storage")

    if index is None:
        logging.error("Index loading failed.")
        return "Error: Could not load the index. Please build it first using rag_pipeline.py."
    logging.info("Index loaded successfully.")

    # 2. Create a query engine from the index
    logging.info("Creating query engine...")
    query_engine = index.as_query_engine()
    logging.info("Query engine ready.")

    # 3. Query the engine
    logging.info(f"Querying with: '{question}'")
    try:
        # --- Added Logging ---
        logging.info("Sending query to LLM...") # Log immediately before the blocking call
        response = query_engine.query(question)
        logging.info("LLM processing finished. Received response object.") # Log immediately after
        # Optional: Log raw response details at DEBUG level
        logging.debug(f"Raw response type: {type(response)}")
        logging.debug(f"Raw response content: {response}")
        # ---------------------

        # Process the response
        answer_text = str(response.response).strip()
        # Llama 3.1 also uses <|eot_id|>
        if answer_text.endswith("<|eot_id|>"):
              answer_text = answer_text[:-len("<|eot_id|>")].strip()
        logging.info("Successfully processed LLM response.") # Log successful processing
        return answer_text
    except Exception as e:
        logging.error(f"An error occurred during querying: {e}", exc_info=True)
        return f"Error during query: {e}"

# --- Main execution block for direct testing (Keep as is) ---
if __name__ == "__main__":
    logging.info("Starting QA System with Llama 3.1 LLM...")
    user_question = input("Ask a question about neural networks (e.g., 'What is ResNet?'): ")
    if user_question:
        answer = answer_nn_question(user_question)
        print("\nSynthesized Answer (Llama 3.1):")
        print(answer)
    else:
        print("No question asked.")