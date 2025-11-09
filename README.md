# Local RAG Document Q&A System

A privacy-first document question-answering system that runs entirely on your local machine. Upload documents, ask questions, and get accurate answers with source citations—without sending your data to any cloud service.

## Why This Exists

Most RAG tutorials use cloud APIs like OpenAI, which means your documents leave your machine. This implementation keeps everything local using Ollama for inference and ChromaDB for vector storage.

## Features

- **Two modes for comparison:**
  - Direct LLM Mode: Ask questions without document context
  - RAG Mode: Get answers grounded in your uploaded documents
- **Complete privacy**: No data leaves your machine
- **Source citations**: See exactly where answers come from

## Tech Stack

- **Ollama**: Local LLM inference
- **ChromaDB**: Vector database for semantic search
- **Sentence-Transformers**: Document embeddings (all-MiniLM-L6-v2)
- **Streamlit**: Web interface

## Installation

# 1. Clone the repo
git clone https://github.com/LocalAIGuide/LocalRag-Course.git
cd LocalRag-Course
```

2. Create a Conda environment:
```bash
   conda env create -f environment.yml
   conda activate local-rag
```

3. Install and run Ollama (see ollama.ai for your platform):
```bash
   ollama pull llama3.2:3b
```

4. Run the app:
```bash
   streamlit run LocalLLM-App1-QA.py
```

## Usage

1. Upload a document (txt, pdf, etc.)
2. Try asking a question in Direct LLM mode
3. Switch to RAG mode and ask the same question
4. Compare the results

## About

This is part of an upcoming Udemy course on building privacy-first RAG systems. Want to learn more?

**Join the email list**: https://localaiguide.kit.com/2fa94df8c1

(Check your spam folder for the welcome email and mark it as "not spam" 
so you don't miss future updates)

## License

MIT
