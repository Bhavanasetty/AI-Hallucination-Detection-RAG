# AI Hallucination Detection using Retrieval-Augmented Generation (RAG)

## Overview

This project focuses on detecting and mitigating hallucinations in Large Language Models (LLMs) using Retrieval-Augmented Generation (RAG). The system enhances factual reliability by grounding model responses in externally retrieved context and evaluating the generated outputs against reference answers.


## Problem Statement

Large Language Models often generate responses that are fluent but factually incorrect, a phenomenon known as hallucination. This project addresses both intrinsic and extrinsic hallucinations by incorporating retrieval-based grounding and evaluating improvements over a baseline model without retrieval.


## Approach

The system implements three configurations:

* **Baseline Model**: Generates answers using FLAN-T5 without external knowledge
* **Semantic RAG**: Uses FAISS with dense embeddings for context retrieval
* **Hybrid RAG**: Combines semantic (FAISS) and lexical (BM25) retrieval using weighted fusion

The generated responses are compared against reference answers to determine factual consistency.


## Methodology

### 1. Data Preparation

* Constructed a dataset of question-answer pairs across multiple domains
* Each sample includes query, context, and reference answer

### 2. Retrieval Mechanism

* **Semantic Retrieval**: SentenceTransformers with FAISS index
* **Lexical Retrieval**: BM25 for exact term matching
* **Hybrid Retrieval**: Weighted combination of both methods

### 3. Generation

* Model: FLAN-T5
* Input: Retrieved context + query
* Output: Generated answer grounded in context

### 4. Evaluation

* Cosine Similarity
* Precision, Recall, F1 Score
* Hallucination Detection (threshold-based classification)


## Results

* Hallucination rate reduced from 50% (baseline) to 25% (RAG)
* Significant improvement in semantic similarity
* F1 score improved by approximately 185%
* Retrieval-based grounding improved factual accuracy across domains


## Tech Stack

* Python
* Hugging Face Transformers
* SentenceTransformers
* FAISS
* BM25 (rank-bm25)
* Scikit-learn
* NumPy
* Matplotlib


## Repository Structure

* `rag_hallucination_detection.py` – Main implementation script
* `hallucination_detection_rag.ipynb` – Notebook version
* `hallucination_report.docx` – Detailed research report


## How to Run

### Install dependencies

pip install -r requirements.txt

### Run the script

python rag_hallucination_detection.py

### Run the notebook

colab notebook


## Key Contributions

* Designed and implemented a complete RAG-based hallucination detection system
* Integrated semantic and lexical retrieval for improved context grounding
* Developed evaluation pipeline for measuring hallucination reduction
* Conducted comparative analysis between baseline, semantic RAG, and hybrid RAG


## Future Work

* Expand dataset size for robust evaluation
* Integrate larger and more advanced language models
* Improve hallucination detection using NLI-based methods
* Deploy as an interactive application


## Author

Bhavana B N
M.Sc Data Science and Business Analytics
