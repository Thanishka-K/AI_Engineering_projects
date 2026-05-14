# Project 6: Context-Aware PDF Chatbot (RAG)
This experiment implements Retrieval-Augmented Generation (RAG) to create a context-aware chatbot. Unlike standard LLMs that rely solely on pre-trained data, this system "reads" uploaded PDF documents and retrieves specific information to provide accurate, fact-based answers.

​##Technical Stack
* ​Orchestration: LangChain
* ​Model: Gemini 1.5 Flash (via Google Generative AI)
* ​Vector Database: ChromaDB
​* Frontend: Streamlit

## ​Architecture & Workflow
* ​Document Ingestion: The PDF is loaded and broken into smaller chunks using a RecursiveCharacterTextSplitter.
* ​Embedding Generation: Each chunk is converted into a high-dimensional vector using GoogleGenerativeAIEmbeddings.
* ​Vector Storage: These embeddings are stored in ChromaDB, allowing for semantic search based on mathematical similarity.
* ​Query Retrieval: When a user asks a question, the system finds the most relevant chunks in the database.
* ​Response Generation: The LLM receives the question and the retrieved text chunks as context to generate a precise answer.
