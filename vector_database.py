from langchain_community.vectorstores import Qdrant


from dotenv import load_dotenv
import os
load_dotenv()



# function for vector database 

def vector_database(docs,embeddings):
    doc_store = Qdrant.from_documents(
    docs, 
    embeddings, 
    url="https://75a92d28-2295-48d3-a46d-692dea422d9a.us-east4-0.gcp.cloud.qdrant.io:6333", 
    api_key=os.getenv('DB_API_KEY'), 
    collection_name="my_collection"
)
    return doc_store