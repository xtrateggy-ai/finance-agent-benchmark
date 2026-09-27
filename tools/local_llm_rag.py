#!/usr/bin/env python3
"""
Question-Answering LLM Extractor - The RIGHT approach
LLM receives the QUESTION and answers it semantically from SEC filings
"""

import os
import sys
import json
#import faiss
from typing import Dict, List, Any, Optional
from pathlib import Path
import asyncio

# LlamaIndex imports
from llama_index.core import (
    Settings, 
    VectorStoreIndex, 
    Document,
    StorageContext
)

# === GPU-accelerated embeddings (works on GTX 1050 Ti + CUDA 12.6) ===
import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core import Document
#from llama_index.embeddings.llama_cpp import LlamaCPPEmbedding
#from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb


os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Limit CPU count to prevent leaks



class QuestionAnsweringExtractor:
    """
    Question-first LLM extraction for SEC filings.
    
    This is the CORRECT approach:
    1. White agent asks a question
    2. Green agent passes question + filings to LLM
    3. LLM semantically understands and answers
    
    """
    
    def __init__(
        self, 
        #model_path: str = "models/qwen2.5-7b-instruct-q5_k_m.gguf",
        #model_path: str = "models/llama-3.2-3b-instruct-q5_k_m.gguf",
        model_path: str | None = None,
        use_local_llm_gpu: bool = False
    ):
        print("[INIT] Loading local LLM...")
            # Resolve model path (ENV > argument > default)
        model_path = (
            os.getenv("LOCAL_LLM_MODEL_PATH")
            or model_path
            or "models/llama-3.2-1b-instruct-q4_k_m.gguf"
        )

        # Optional safety check
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Local LLM model not found at: {model_path}"
            )
        # Initialize LLM
        self.llm = LlamaCPP(
            model_path=model_path,
            temperature=0.0,           # or 0.1 if you like slight variation
            max_new_tokens=256,        # more than enough for public.csv
            #context_window=32769,                
            context_window=4096,       # using only 4k to speed up processing 
            verbose=False,
            model_kwargs={
                "n_gpu_layers": 15 if use_local_llm_gpu else 0,
                #"n_ctx": 32769, #8192,  # llama.cpp parameter (inside model_kwargs)
                "n_ctx": 4096,    # using only 4k to speed up processing
                "n_batch": 128,   # 🚨 IMPORTANT
                "n_threads": 4,   # Use 8 CPU threads for better performance
                "main_gpu": 15 if use_local_llm_gpu else 0,  
                "tensor_split": None,  # Not needed for single GPU
            }
        )       
        
        # Use a fast, high-quality model that fits in 4 GB VRAM
        #EMBED_MODEL = "BAAI/bge-small-en-v1.5"        # 4× faster + better than all-MiniLM
        EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # fallback if you want
        
        # Initialize embeddings for RAG
        self.embed_model = HuggingFaceEmbedding(
            model_name=EMBED_MODEL,
            #device="cuda" if torch.cuda.is_available() else "cpu",
            device="cpu",
            embed_batch_size=64  # safe for 4 GB card  
        )
        
        # Set global settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50
                
        # Cache for indexed filings
        self.index_cache = {}
        
        print("[INIT] ✅ Ready to answer questions")


    def __del__(self):
        """Cleanup method to prevent resource leaks"""
        try:
            # Clean up LLM
            if hasattr(self, 'llm'):
                del self.llm
            
            # Clean up embeddings
            if hasattr(self, 'embed_model'):
                del self.embed_model
            
            # Clear index cache
            if hasattr(self, 'index_cache'):
                self.index_cache.clear()
        except Exception as e:
            print(f"[LOCAL_LLM_RAG] Cleanup warning: {e}")
            

    async def answer_question_with_rag(
        self,
        question: str,
        data: List[Document],
        company: str,
        use_disk_cache: bool = False
    ) -> str:
        """
        Use RAG when question requires comparing multiple filings.
        
        NOW WITH ACTUAL DISK CACHING:
        - Checks if embeddings already exist in ChromaDB
        - Only re-embeds if collection is empty
        - Saves ~90% of processing time on reruns
        """
        try:
            print(f"[LOCAL_LLM_RAG] Question=[{question[:80]}]")        
                 
            # ================================================================
            # STEP 1: Setup ChromaDB (in-memory or persistent)
            # ================================================================
            if use_disk_cache:
                # Use CIK for cache key (most reliable)
                cik = data[0].metadata.get("cik") if data else None
                if cik:
                    cache_path = f"./chroma_db/cik_{cik}"
                else:
                    # Fallback to company name
                    cache_path = f"./chroma_db/{company.lower().replace(' ', '_')}"
                
                chroma_client = chromadb.PersistentClient(path=cache_path)
                print(f"[LOCAL_LLM_RAG] Using persistent ChromaDB: {cache_path}")
            else:
                chroma_client = chromadb.EphemeralClient()
                print(f"[LOCAL_LLM_RAG] Using in-memory ChromaDB (no cache)")
            
            # ================================================================
            # STEP 2: Get or create collection
            # ================================================================
            collection_name = "sec_filings"
            chroma_collection = chroma_client.get_or_create_collection(name=collection_name)
    
            # Set up vector store
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # ================================================================
            # STEP 3: Check if collection already has embeddings (KEY FIX!)
            # ================================================================
            existing_count = chroma_collection.count()
            
            if existing_count > 0:
                # ✅ EMBEDDINGS ALREADY EXIST - Just load the index
                print(f"[LOCAL_LLM_RAG] ✅ Found {existing_count} existing embeddings in cache")
                print(f"[LOCAL_LLM_RAG] Skipping re-embedding (saves ~90% time)")
                
                # Load existing index from storage
                index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    storage_context=storage_context
                )
            else:
                # ❌ COLLECTION IS EMPTY - Need to embed documents
                print(f"[LOCAL_LLM_RAG] Collection empty, embedding {len(data)} documents...")
                print(f"[LOCAL_LLM_RAG] (This will take a while...)")
                
                # Create new index with embeddings
                index = VectorStoreIndex.from_documents(
                    data,
                    storage_context=storage_context,
                    show_progress=True
                )
                
                print(f"[LOCAL_LLM_RAG] ✅ Embeddings saved to: {cache_path if use_disk_cache else 'memory'}")
                
            print(f"[LOCAL_LLM_RAG] After VectorStoreIndex setup")  
            
            # ================================================================
            # STEP 4: Query the index
            # ================================================================
            query_engine = index.as_query_engine(
                similarity_top_k=2,  # Reduced from 6 
                response_mode="compact",  # Changed from "simple_summarize"
                system_prompt="""You are a precise SEC filings analyst.
            
            RULES:
            1. Answer ONLY using information from the provided documents
            2. For list questions (e.g., "list all board nominees"), provide a complete comma-separated list
            3. For numerical questions, provide the exact number with units
            4. For yes/no questions, start with YES or NO, then explain briefly
            5. If information is not in the documents, say "Information not found in filings"
            6. Be concise but complete
            
            EXAMPLES:
            Q: "List all board nominees"
            A: "James Miller, Michael Elich, Anthony Harris, Jon Roberts, Kimberly Lody"
            
            Q: "What was the revenue?"
            A: "$5.2 billion"
            
            Q: "Did they beat guidance?"
            A: "YES. Pre-tax margin was 10.8%, beating guidance of 10.6% by 20 basis points"
            """
            )
            
            print(f"[LOCAL_LLM_RAG] After index.as_query_engine()") 
            
            # Query and get response
            response = query_engine.query(question)
            
            # Handle async response if needed
            if asyncio.iscoroutine(response):
                response = await response
            
            answer = str(response).strip()
            
            print(f"[LOCAL_LLM_RAG] Returning answer={answer}")
            return answer
    
        except Exception as e:
            print(f"[LOCAL_LLM_RAG] ⚠️ RAG failed: {e}", file=sys.stderr)
            return "Error: RAG processing failed."    
    
"""    
    # ================================================================
    # Answer Question Across Multiple Filings (RAG)
    # ================================================================
    async def answer_question_with_rag(
        self,
        question: str,
        #data: str,
        data: List[Document],
        company: str,
        use_disk_cache: bool = False
    ) -> str:
        
        # Use RAG when question requires comparing multiple filings.
        
        # Use cases:
        # - "How did Netflix ARPPU change from 2019 to 2024?"
        # - "Compare TJX margins in Q4 2024 vs Q4 2023"
        # - "When was the Nippon Steel merger first announced?"
        
        # Args:
        #     question: User question requiring multi-filing analysis
        #     filings: List of {"text": str, "date": str, "form": str}
        #     company: Company name for context
        
        # Returns:
        #     Comprehensive answer synthesized from multiple filings
        
        try:
            
            print(f"[LOCAL_LLM_RAG] Question=[{question[:80]}]")        
                 
            # Create in-memory or persistent Chroma client
            if use_disk_cache:
                # Use CIK from document metadata if available
                cik = data[0].metadata.get("cik") if data else None
                if cik:
                    cache_path = f"./chroma_db/cik_{cik}"
                else:
                    # Fallback to company name
                    cache_path = f"./chroma_db/{company.lower().replace(' ', '_')}"
                
                chroma_client = chromadb.PersistentClient(path=cache_path)
                print(f"[LOCAL_LLM_RAG] Using persistent ChromaDB: {cache_path}")
            else:
                chroma_client = chromadb.EphemeralClient()
                
    	    # Create or load collection
            chroma_collection = chroma_client.get_or_create_collection(name="sec_filings")

            # Set up vector store
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            index = VectorStoreIndex.from_documents(
                data,
                storage_context=storage_context,
                show_progress=True
            )
                
            print(f"[LOCAL_LLM_RAG] After VectorStoreIndex.from_documents()")  
            
            # Query engine
            query_engine = index.as_query_engine(similarity_top_k=2,  # Reduced from 6 
                            #response_mode="simple_summarize",
                            response_mode="compact",  # Changed from "simple_summarize"
                        #     system_prompt=<place triple quotes here> You are a precise SEC filings analyst.
                        
                        # RULES:
                        # 1. Answer ONLY using information from the provided documents
                        # 2. For list questions (e.g., "list all board nominees"), provide a complete comma-separated list
                        # 3. For numerical questions, provide the exact number with units
                        # 4. For yes/no questions, start with YES or NO, then explain briefly
                        # 5. If information is not in the documents, say "Information not found in filings"
                        # 6. Be concise but complete
                        
                        # EXAMPLES:
                        # Q: "List all board nominees"
                        # A: "James Miller, Michael Elich, Anthony Harris, Jon Roberts, Kimberly Lody"
                        
                        # Q: "What was the revenue?"
                        # A: "$5.2 billion"
                        
                        # Q: "Did they beat guidance?"
                        # A: "YES. Pre-tax margin was 10.8%, beating guidance of 10.6% by 20 basis points"
                        # <place triple quotes here>
            )
            
            print(f"[LOCAL_LLM_RAG] After index.as_query_engine()") 
            
            # ⚠️ THIS IS THE PROBLEM - query is NOT awaited properly!
            response = query_engine.query(question)
            
            # FIX: Make sure to await if it returns a coroutine
            if asyncio.iscoroutine(response):
                response = await response
            
            answer = str(response).strip()
            
            #print(f"[LOCAL_LLM_RAG] Returning answer={answer[:100]}")
            print(f"[LOCAL_LLM_RAG] Returning answer={answer}")
            return answer

        except Exception as e:
            print(f"[LOCAL_LLM_RAG] ⚠️ RAG failed: {e}", file=sys.stderr)
            # ← FIX 1: return string, NOT dict
            return "Error: RAG processing failed."
"""



# ================================================================
# Usage Examples
# ================================================================

async def demo():
    """Show question-answering approach"""
    
    extractor = QuestionAnsweringExtractor(
        #model_path="models/qwen2.5-7b-instruct-q5_k_m.gguf"
        #model_path="models/llama-3.2-3b-instruct-q5_k_m.gguf"
        model_path="models/llama-3.2-1b-instruct-q4_k_m.gguf"
    )
    
    # ────────────────────────────────────────────────────────────
    # Example 1: Board Nominees Question
    # ────────────────────────────────────────────────────────────
    print("=" * 60)
    print("EXAMPLE 1: Board Nominees")
    print("=" * 60)
    
    filing = {
        "text": """
        PROPOSAL 1 - ELECTION OF DIRECTORS
        
        The Board has nominated the following eight individuals for election:
        James C. Miller, Michael L. Elich, Anthony Harris, Jon L. Roberts,
        Kimberly S. Lody, Gary E. Kramer, Vincent R. Price, Sarah T. Wang
        """,
        "form": "DEF 14A",
        "date": "2024-04-15"
    }
    
    question = "List all board nominees for BBSI in 2024"
    
    # answer = await extractor.answer_question(
    #     question=question,
    #     filing_text=bbsi_filing["text"],
    #     filing_metadata={
    #         "form": "DEF 14A",
    #         "date": "2024-04-15",
    #         "company": "BBSI"
    #     }
    # )
    data = [Document(
        text=filing["text"],
        metadata={
            "form": filing["form"],
            "date": filing["date"],
            "company": "BBSI"
            }
    )]
    answer = await extractor.answer_question_with_rag(
        question=question,
        data=data,
        company="BBSI"
    )
    
    print(f"Q: {question}")
    print(f"A: {answer}")
    
    # ────────────────────────────────────────────────────────────
    # Example 2: Pre-tax Margin Question
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Pre-tax Margin Beat/Miss")
    print("=" * 60)
    
    filing = {
        "text": """
        FOURTH QUARTER RESULTS
        
        Pre-tax profit margin was 10.8%, 20 basis points above the 
        high end of our plan of 10.6%. This beat was driven by stronger 
        than expected merchandise margins.
        """,
        "form": "8-K",
        "date": "2025-02-26"
    }
    
    question = "Did TJX beat its Q4 FY 2025 pre-tax margin guidance? Express result as BPS difference."
    
    # answer = await extractor.answer_question(
    #     question=question,
    #     filing_text=tjx_filing["text"],
    #     filing_metadata={
    #         "form": "8-K",
    #         "date": "2025-02-26",
    #         "company": "TJX"
    #     }
    # )
    data = [Document(
        text=filing["text"],
        metadata={
            "form": filing["form"],
            "date": filing["date"],
            "company": "TJX"
            }
    )]
    answer = await extractor.answer_question_with_rag(
        question=question,
        data=data,
        company="TJX"
        )
    
    print(f"Q: {question}")
    print(f"A: {answer}")
    
    # ────────────────────────────────────────────────────────────
    # Example 3: Multi-Filing Comparison (RAG)
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Multi-Year ARPPU Comparison")
    print("=" * 60)
    
    filings = [
        {
            "text": "2019 ARPPU was $11.23 with 167M members",
            "date": "2020-02-15",
            "form": "10-K"
        },
        {
            "text": "2024 ARPPU was $15.49 with 280M members",
            "date": "2025-02-15",
            "form": "10-K"
        }
    ]
    
    question = "How did Netflix ARPPU change from 2019 to 2024?"
    data = [
        Document(
            text=f["text"],
            metadata={"date": f["date"], "form": f["form"], "company": "Netflix"}
        )
        for f in filings
    ]
    
    answer = await extractor.answer_question_with_rag(
        question=question,
        data=data,
        company="Netflix"
    )
    
    print(f"Q: {question}")
    print(f"A: {answer}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo())
