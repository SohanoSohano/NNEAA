# src/qa_system.py

from src.rag_pipeline import load_faiss_index, DEFAULT_EMBED_MODEL

from llama_index.core import Settings, PromptTemplate
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.postprocessor import SentenceTransformerRerank

import logging
import torch
import os
from rich.console import Console
from rich.markdown import Markdown
from dotenv import load_dotenv

# Configuration
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('llama_index.core.chat_engine').setLevel(logging.INFO)
logging.getLogger('llama_index.core.llms').setLevel(logging.INFO)
logging.getLogger(__name__).setLevel(logging.INFO)

# Model Names
EMBEDDING_MODEL_NAME = DEFAULT_EMBED_MODEL
LLM_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

# Global Settings Configuration
logging.info("Configuring global LlamaIndex settings...")
try:
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
    logging.info(f"Using embedding model: {Settings.embed_model.model_name}")

    logging.info(f"Setting up LLM: {LLM_MODEL_NAME}")
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logging.warning("HF_TOKEN environment variable not set. LLM loading might fail.")

    Settings.llm = HuggingFaceLLM(
        model_name=LLM_MODEL_NAME,
        tokenizer_name=LLM_MODEL_NAME,
        context_window=131072,
        max_new_tokens=512,
        model_kwargs={},
        generate_kwargs={"temperature": 0.7, "do_sample": True},
        device_map="auto",
    )
    logging.info(f"LLM '{LLM_MODEL_NAME}' configured successfully.")
except Exception as e:
    logging.error(f"Failed during Global Settings configuration: {e}", exc_info=True)
    Settings.llm = None

# System Prompt
system_prompt = """
You are an expert Q&A assistant specialized in neural network architectures and AI research. 
Your goal is to answer the user's query accurately and informatively based *only* on the provided context information from research papers. 
Follow these guidelines strictly:

1. Base your entire answer *only* on the text provided in the 'Context information' section. Do not use any external knowledge.
2. If the context doesn't contain the answer, clearly state 'Based on the provided documents, I cannot answer that question.' Do not attempt to guess.
3. If the user's query is ambiguous (e.g., 'tell me about transformers'), ask a specific clarifying follow-up question (e.g., 'Are you asking about the Transformer architecture in NLP, electrical transformers, or something else?') before providing an answer.
4. When providing information derived from the context, cite the source document(s) by mentioning their title or filename (e.g., 'According to the paper titled "Attention Is All You Need"...' or 'The document xyz.pdf mentions...'). Use the 'title' or 'file_name' metadata provided with the context chunks.
5. Structure complex answers clearly using bullet points or numbered steps.
6. Maintain a helpful, expert tone suitable for discussing research topics.
7. Remember the conversation history to understand follow-up questions from the user.
"""

# Reranker Initialization
try:
    reranker = SentenceTransformerRerank(
        model=RERANKER_MODEL_NAME,
        top_n=3
    )
    logging.info(f"Reranker '{RERANKER_MODEL_NAME}' initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize Reranker: {e}", exc_info=True)
    reranker = None

# Main Chat Loop
if __name__ == "__main__":
    if Settings.llm is None:
        print("\nERROR: LLM failed to initialize during setup. Cannot start chat system.")
    elif reranker is None:
        print("\nERROR: Reranker failed to initialize. Cannot start chat system.")
    else:
        console = Console()
        console.print("[bold cyan]Starting Enhanced Neural Network RAG Chat System...[/bold cyan]")
        console.print(f"[cyan]LLM: {LLM_MODEL_NAME}[/cyan]")
        console.print(f"[cyan]Embedding: {EMBEDDING_MODEL_NAME}[/cyan]")
        console.print(f"[cyan]Reranker: {RERANKER_MODEL_NAME}[/cyan]")

        console.print("[cyan]Loading vector index...[/cyan]")
        index = load_faiss_index(persist_dir="storage", embed_model_name=EMBEDDING_MODEL_NAME)
        if index is None:
            console.print("[bold red]ERROR: Could not load the index. Please build it first using 'python src/rag_pipeline.py'.[/bold red]")
        else:
            console.print("[bold green]Index loaded successfully.[/bold green]")

            memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

            chat_engine = index.as_chat_engine(
                chat_mode="context",
                memory=memory,
                system_prompt=system_prompt,
                similarity_top_k=10,
                node_postprocessors=[reranker]
            )

            console.print("[bold green]Chat engine with Reranker ready.[/bold green]")
            console.print("[cyan]Type 'exit' or 'quit' to end the chat.[/cyan]\n")

            while True:
                try:
                    user_input = input("You: ")
                    if user_input.lower() in ["exit", "quit"]:
                        console.print("[bold cyan]Exiting chat. Goodbye![/bold cyan]")
                        break
                    if not user_input:
                        continue

                    console.print(f"[grey50]Processing (Retrieval, Reranking, LLM)...[/grey50]", end='\r')
                    streaming_response = chat_engine.stream_chat(user_input)
                    full_response_text = ""
                    console.print(f"[bold green]Assistant:[/bold green] ", end="")
                    for token in streaming_response.response_gen:
                        print(token, end="", flush=True)
                        full_response_text += token
                    print("\n")

                    source_nodes = streaming_response.source_nodes
                    if source_nodes:
                        console.print("\n[bold yellow]Sources Considered (Post-Reranking):[/bold yellow]")
                        seen_sources = set()
                        for node in source_nodes:
                            title = node.metadata.get('title', None)
                            file_name = node.metadata.get('file_name', None)
                            display_name = title if title and title != "N/A" else file_name
                            if display_name and display_name not in seen_sources:
                                console.print(f"- {display_name}")
                                seen_sources.add(display_name)
                        if not seen_sources:
                            console.print("- (No specific source titles/filenames found in metadata)")
                    print("")

                except Exception as e:
                    logging.error(f"An error occurred during chat: {e}", exc_info=True)
                    console.print(f"\n[bold red]Error: {e}[/bold red]")
                    console.print("[yellow]Restarting chat engine state might be needed if errors persist.[/yellow]")
