# RAG-Powered Document Q&A System

An AI-powered document question-answering application that allows users to upload a PDF and ask questions about its content. The system uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from the document and generate context-aware answers.

## Features

- Upload and process PDF documents
- Extract and split document text into smaller chunks
- Generate vector embeddings using OpenAI
- Store and retrieve document embeddings using ChromaDB
- Find document sections most relevant to a user's question
- Generate answers based on retrieved document context
- Interactive interface built with Streamlit

## How It Works

1. **Upload a PDF**  
   The application extracts text from each page using `pdfplumber`.

2. **Process the Document**  
   Extracted text is divided into smaller chunks that can be searched efficiently.

3. **Create Embeddings**  
   OpenAI's embedding model converts each chunk into a numerical vector representing its meaning.

4. **Store the Embeddings**  
   ChromaDB stores the document chunks and their corresponding embeddings.

5. **Ask a Question**  
   The user's question is also converted into an embedding.

6. **Retrieve Relevant Information**  
   ChromaDB compares the question with the document chunks and retrieves the most relevant sections.

7. **Generate an Answer**  
   The retrieved context is provided to an OpenAI model, which generates an answer based on the document.

## Tech Stack

- Python
- OpenAI API
- ChromaDB
- Streamlit
- pdfplumber
- python-dotenv

## Installation

Clone the repository:

git clone https://github.com/nidhinandakumar13/RAG-Powered-Document-Q-A-Systems.git

Install the required packages:

pip install -r requirements.txt

Create a `.env` file and add your OpenAI API key:

OPENAI_API_KEY=your_api_key_here

Run the application:

streamlit run main.py

## Example Use Case

A user can upload a research paper, report, textbook chapter, or other PDF and ask questions such as:

- "What are the main findings of this document?"
- "Summarize the key recommendations."
- "What does the document say about machine learning?"

The application retrieves relevant sections of the PDF and generates an answer grounded in the document's content.

## What I Learned

Through this project, I gained hands-on experience with:

- Retrieval-Augmented Generation (RAG)
- Vector embeddings and semantic search
- Vector databases
- Working with the OpenAI API
- PDF text extraction and processing
- Building interactive AI applications with Streamlit

## Future Improvements

- Support multiple documents at once
- Add conversation history
- Display source sections used to generate each answer
- Support additional file formats
- Improve document chunking and retrieval
