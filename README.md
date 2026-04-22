# AI Hallucination Detection using RAG with Hybrid Retrieval

## Overview
This project focuses on mitigating hallucinations in Large Language Models (LLMs) using Retrieval-Augmented Generation (RAG) combined with hybrid retrieval techniques (semantic + lexical).

The system evaluates how grounding LLM outputs with retrieved context improves factual accuracy and reduces hallucinated responses.

## Key Features
- Detects hallucinations in LLM-generated responses
- Implements **Semantic Retrieval (FAISS)**
- Implements **Lexical Retrieval (BM25)**
- Hybrid retrieval fusion (α = 0.6, β = 0.4)
- Uses **FLAN-T5** for answer generation
- Evaluation using:
  - Cosine Similarity
  - Precision, Recall, F1 Score
  - Hallucination Detection Threshold


## Tech Stack
- Python
- Hugging Face Transformers (FLAN-T5)
- SentenceTransformers
- FAISS
- BM25 (rank-bm25)
- Scikit-learn
- NumPy, Matplotlib


## Methodology
1. Query is given as input  
2. Relevant documents retrieved using:
   - Semantic search (FAISS)
   - Lexical search (BM25)
3. Hybrid retrieval combines both  
4. LLM generates response using retrieved context  
5. Output is evaluated for hallucination  


## Results
- Hallucination rate reduced from **50% → 25%**
- F1 Score improved by **185%**
- Significant improvement in factual grounding using RAG


## Files in Repository
- `AI_Hallucination_Detection_with_RAG.ipynb` → Implementation
- `Hallucination_Report.docx` → Detailed research report

---

## ▶️ How to Run
```bash
pip install -r requirements.txt
