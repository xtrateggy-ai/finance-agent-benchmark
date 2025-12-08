#!/usr/bin/env python3
"""
Question-Answering LLM Extractor - The RIGHT approach
LLM receives the QUESTION and answers it semantically from SEC filings
"""

import os
import sys
import json
import faiss
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
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document
#from llama_index.embeddings.llama_cpp import LlamaCPPEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore


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
        model_path: str = "models/llama-3.2-1b-instruct-q4_k_m.gguf",
        use_local_llm_gpu: bool = True
    ):
        print("[INIT] Loading local LLM...")
        
        # Initialize LLM
        """
        self.llm = LlamaCPP(
            model_path=model_path,
            temperature=0.1,
            max_new_tokens=512,
            context_window=32768,
            model_kwargs={"n_gpu_layers": 0 if not use_gpu else 35},
            verbose=False,
        )
        """
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
                "n_threads": 4,   # Use 8 CPU threads for better performance
                "main_gpu": 0,  # Use first GPU (your Intel iGPU)
                "tensor_split": None,  # Not needed for single GPU
            }
        )       
        
        # Initialize embeddings for RAG
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        #Settings.embed_model = LlamaCPPEmbedding(
        #    model_path="models/nomic-embed-text-v1.5.Q5_K_M.gguf",
        #    n_ctx=8192,
        #)
        
        # Set global settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50
                
        # Cache for indexed filings
        self.index_cache = {}
        
        print("[INIT] ✅ Ready to answer questions")
    
    
    # ================================================================
    # Answer Question Across Multiple Filings (RAG)
    # ================================================================
    async def answer_question_with_rag(
        self,
        question: str,
        #data: str,
        data: List[Document],
        company: str
    ) -> str:
        """
        Use RAG when question requires comparing multiple filings.
        
        Use cases:
        - "How did Netflix ARPPU change from 2019 to 2024?"
        - "Compare TJX margins in Q4 2024 vs Q4 2023"
        - "When was the Nippon Steel merger first announced?"
        
        Args:
            question: User question requiring multi-filing analysis
            filings: List of {"text": str, "date": str, "form": str}
            company: Company name for context
        
        Returns:
            Comprehensive answer synthesized from multiple filings
        """
        try:
            
            #print(f"[LOCAL_LLM_RAG] Question=[{question[:80]}]    Data=[{data[0]['text'][:80]}]")        
            print(f"[LOCAL_LLM_RAG] Question=[{question[:80]}]")        
             
            # Create FAISS vector store
            faiss_index = faiss.IndexFlatL2(384) # 384 dimensions for all-MiniLM-L6-v2
                                                 # 768 dimensions for nomic-embed-text-v1.5.Q5_K_M.gguf
                                                 
            print(f"[LOCAL_LLM_RAG] After faiss.IndexFlatL2(384)") 
            
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            print(f"[LOCAL_LLM_RAG] After FaissVectorStore()") 
            
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            print(f"[LOCAL_LLM_RAG] After StorageContext.from_defaults()") 
            
            # index = VectorStoreIndex.from_documents(
            #     data,
            #     storage_context=storage_context,
            #     show_progress=True
            # )
            
            # add cache index
            cache_key = hashlib.sha1((company + question[:64]).encode("utf-8")).hexdigest()

            # Try to reuse an existing index for this company
            if cache_key in self.index_cache:
                index = self.index_cache[cache_key]
                print("[LOCAL_LLM_RAG] Using cached index")
            else:
                index = VectorStoreIndex.from_documents(
                    data,
                    storage_context=storage_context,
                    show_progress=True
                )
                self.index_cache[cache_key] = index
                print("[LOCAL_LLM_RAG] Cached new index")

            
            print(f"[LOCAL_LLM_RAG] After VectorStoreIndex.from_documents()")  
            
            # Query engine
            query_engine = index.as_query_engine(similarity_top_k=2,  # Reduced from 6 
                                                 response_mode="simple_summarize",
                                                 system_prompt="""You are a precise SEC filings analyst. 
    Extract exact facts only from the provided documents. 
    For multi-year questions, list values chronologically with dates. 
    Never add information not present in the filings."""
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