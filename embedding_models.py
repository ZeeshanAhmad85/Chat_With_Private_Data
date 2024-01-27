# from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings


def apply_embedding_model1():
    # Load the model
    embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    )

    # Return the embeddings
    return embeddings

def apply_embedding_model2():
    embeddings=HuggingFaceEmbeddings(
        model_name="intfloat/e5-large-v2",
    )

    return embeddings