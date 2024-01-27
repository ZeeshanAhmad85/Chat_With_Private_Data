# Chat_With_Private_Data


## ChatBot Project

## Overview

This project implements a ChatBot system using Streamlit, language models (GPT-3.5 Turbo), and a vector database for document retrieval. The system allows users to upload documents in various formats (PDF, DOCX, ZIP), processes the content, applies text splitting, generates embeddings, stores them in a vector database, and provides conversational responses to user queries.

## Getting Started

### Prerequisites

- Python 3.6+
- [Streamlit](https://streamlit.io/)
- [Hugging Face Transformers](https://huggingface.co/models)
- [Langchain Community Library](https://github.com/Langchain/community)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/your-repo.git
cd your-repo

# Install dependencies
pip install -r requirements.txt

## Create a .env file in the project root and add the following:
makefile
API_KEY=your_gpt_api_key
DB_API_KEY=your_vector_db_api_key
Replace your_gpt_api_key and your_vector_db_api_key with your actual API keys.

## Set up the vector database URL in your code.
Usage
Run the Streamlit app:

bash
Copy code
streamlit run your_script.py


## Features
Document upload and processing (PDF, DOCX, ZIP)
Text splitting options (CharacterTextSplitter, RecursiveCharacterTextSplitter)
Embedding models (Hugging Face Transformers)
Vector database integration (Qdrant)
Conversational responses using GPT-3.5 Turbo
File Structure
your_script.py: Main Streamlit application script.
data_loader.py: Document processing and loading functions.
text_splitters.py: Text splitting functions.
embedding_models.py: Embedding model functions.
vector_database.py: Vector database functions.
