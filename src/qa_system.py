# src/qa_system.py

# Import the function to load the index
from src.rag_pipeline import load_faiss_index
# Core LlamaIndex imports
from llama_index.core import Settings, PromptTemplate
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondenseQuestionChatEngine, ContextChatEngine # Or other chat engine types

# Embedding and LLM imports
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
# Standard library
import logging
import torch
# Rich for formatted printing
from rich.console import Console
from rich.markdown import Markdown

# Configure logging (set higher level to reduce noise during chat)
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
# Set specific loggers if needed, e.g., llama_index
# logging.getLogger('llama_index.core').setLevel(logging.INFO)

# --- Global Settings (Same as before) ---
logging.info("Configuring global settings for LlamaIndex...")
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
logging.info(f"Using embedding model: {Settings.embed_model.model_name}")

llm_model_name = "meta-llama/Llama-3.2-1B-Instruct"
logging.info(f"Setting up LLM: {llm_model_name}")

# --- NEW System Prompt for Chat Engine ---
# Instructing the LLM on conversational behavior, source usage, and follow-ups
system_prompt = (
    "You are an expert Q&A assistant specialized in neural network architectures and development. "
    "Your goal is to answer the user's query accurately and informatively based *only* on the provided context information. "
    "Follow these guidelines:\n"
    "1. Base your entire answer *only* on the text provided in the 'Context information' section. Do not use any prior knowledge.\n"
    "2. If the context doesn't contain the answer, clearly state that the information is not available in the provided documents.\n"
    "3. If the user's query is ambiguous or lacks detail, ask relevant clarifying follow-up questions before attempting a full answer.\n"
    "4. When providing information found in the context, try to mention the source document (e.g., filename if available in metadata) that contains the information.\n"
    "5. Structure answers clearly, using steps or bullet points where appropriate.\n"
    "6. Keep conversation history in mind for relevant follow-up interactions."
)
# Note: The Llama 3 prompt format might be slightly different for system prompts in chat engines.
# We'll pass this via the chat engine's system_prompt argument. LlamaIndex handles formatting.

# --- LLM Initialization (Quantization Disabled) ---
logging.info("Initializing LLM (Quantization Disabled).")
try:
    Settings.llm = HuggingFaceLLM(
        model_name=llm_model_name,
        tokenizer_name=llm_model_name,
        # query_wrapper_prompt is less relevant for chat engine, system_prompt is key
        context_window=131072,
        max_new_tokens=512,
        model_kwargs={},
        generate_kwargs={"temperature": 0.7, "do_sample": True},
        device_map="auto",
    )
    logging.info(f"LLM '{llm_model_name}' configured successfully.")
except Exception as e:
    logging.error(f"LLM Initialization Failed: {e}", exc_info=True)
    Settings.llm = None # Ensure it's None if failed

# --- Main Chat Loop ---
if __name__ == "__main__":
    if Settings.llm is None:
        print("\nERROR: LLM failed to initialize. Cannot start chat system.")
    else:
        console = Console() # For rich printing
        console.print("[bold cyan]Starting Neural Network RAG Chat System...[/bold cyan]")
        console.print(f"[cyan]Using LLM: {llm_model_name}[/cyan]")

        # Load the index
        console.print("[cyan]Loading vector index...[/cyan]")
        index = load_faiss_index(persist_dir="storage")

        if index is None:
            console.print("[bold red]ERROR: Could not load the index. Please build it first.[/bold red]")
        else:
            console.print("[bold green]Index loaded successfully.[/bold green]")

            # Create chat memory
            memory = ChatMemoryBuffer.from_defaults(token_limit=3900) # Adjust token limit as needed

            # Create chat engine (ContextChatEngine uses context + memory)
            chat_engine = index.as_chat_engine(
                chat_mode="context",
                memory=memory,
                system_prompt=system_prompt,
                # Optional: Customize how context is retrieved/used
                # similarity_top_k=5,
            )
            console.print("[bold green]Chat engine ready.[/bold green]")
            console.print("[cyan]Type 'exit' or 'quit' to end the chat.[/cyan]\n")

            # Start conversation loop
            while True:
                try:
                    user_input = input("You: ")
                    if user_input.lower() in ["exit", "quit"]:
                        console.print("[bold cyan]Exiting chat. Goodbye![/bold cyan]")
                        break
                    if not user_input:
                        continue

                    console.print(f"[grey50]Processing using {llm_model_name}...[/grey50]", end='\r')
                    # Use chat engine's stream_chat for interactive feel, or chat for blocking
                    # response = chat_engine.chat(user_input) # Blocking call

                    # Streaming call - prints tokens as they are generated
                    streaming_response = chat_engine.stream_chat(user_input)
                    full_response_text = ""
                    console.print(f"[bold green]Assistant:[/bold green] ", end="")
                    for token in streaming_response.response_gen:
                        print(token, end="", flush=True)
                        full_response_text += token
                    print("\n") # Newline after streaming finishes

                    # Process source nodes after response is complete
                    source_nodes = streaming_response.source_nodes
                    if source_nodes:
                        console.print("\n[bold yellow]Sources Used:[/bold yellow]")
                        seen_files = set()
                        for i, node in enumerate(source_nodes):
                             # Attempt to get filename from metadata
                             file_name = node.metadata.get('file_name', f'Source {i+1}')
                             if file_name not in seen_files:
                                 # Optionally print score: score = node.get_score()
                                 console.print(f"- {file_name}")
                                 seen_files.add(file_name)
                        print("") # Add spacing

                except Exception as e:
                    logging.error(f"An error occurred during chat: {e}", exc_info=True)
                    console.print(f"[bold red]Error: {e}[/bold red]")

