from langchain_community.vectorstores import Qdrant


from dotenv import load_dotenv
import os
load_dotenv()



# function for vector database 

def vector_database(docs,embeddings):
    doc_store = Qdrant.from_documents(
    docs, 
    embeddings, 
    url="YOUR DATABASE URL", 
    api_key=os.getenv('DB_API_KEY'), 
    collection_name="my_collection"
)
    return doc_store
