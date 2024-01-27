import streamlit as st
import sys
from data_loader import process_document,process_zip
# import text_splitters
from text_splitters import apply_char_text_splitter,apply_recursive_text_splitter
# import embedding_models
from embedding_models import apply_embedding_model1,apply_embedding_model2
# from langchain_community.vectorstores import Qdrant
from vector_database import vector_database


from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter


from dotenv import load_dotenv


# Load the environment variables from the .env file
load_dotenv()


from typing import List
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import os



# from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings



# Output parser will split the LLM result into a list of queries
class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")



class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)


output_parser = LineListOutputParser()


QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines.
    Original question: {question}""",
)



llm=ChatOpenAI(
            temperature=0.6,
            api_key = os.getenv('API_KEY'),
            model_name="gpt-3.5-turbo-16k"
            )





# Chain
llm_chain = LLMChain(llm=llm, prompt=QUERY_PROMPT, output_parser=output_parser)

#################################################################################################################################

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Question: \n{question}\n
    Context: \n{context}\n
    Answer:
    """

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

    return chain

#################################################################################################################################
def user_input(user_question,unique_docs):

    # context=" "
    chain = get_conversational_chain()
    
       
    response = chain(
        {
            "input_documents":unique_docs, 
            "question": user_question
        }, 
        return_only_outputs=True
        )

    return response["output_text"]

########################################################################################################################################

# function for text splitter



# def apply_char_text_splitter(pages):
#     # Apply text splitter
#     text_splitter = CharacterTextSplitter(
#         separator="\n\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len,
#         is_separator_regex=False,
#     )

#     return text_splitter.split_documents(pages)

# def apply_recursive_text_splitter(pages):
#     # Apply text splitter
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len,
#         is_separator_regex=False,
#     )

#     return text_splitter.split_documents(pages)

#######################################################################################################################################

# def apply_embedding_model1():
#     # Load the model
#     embeddings = HuggingFaceEmbeddings(
#     model_name="BAAI/bge-small-en-v1.5",
#     )

#     # Return the embeddings
#     return embeddings

# def apply_embedding_model2():
#     embeddings=HuggingFaceEmbeddings(
#         model_name="intfloat/e5-large-v2",
#     )

#     return embeddings

####################################################################################################################################
# function for vector database 

# def vector_database(docs,embeddings):
#     doc_store = Qdrant.from_documents(
#     docs, 
#     embeddings, 
#     url="https://75a92d28-2295-48d3-a46d-692dea422d9a.us-east4-0.gcp.cloud.qdrant.io:6333", 
#     api_key=os.getenv('DB_API_KEY'), 
#     collection_name="my_collection"
# )
#     return doc_store





####################################################################################################################################

# Streamlit frontend
st.title("Chat With Me")
uploaded_files = st.file_uploader("Choose file(s)", type=["pdf", "docx", "zip"], accept_multiple_files=True)
pages=[]
# Check if files were uploaded
if uploaded_files:
    try:
        for file in uploaded_files:
            file_type = file.type

            # Process uploaded files based on file type
            if file_type == "application/pdf":

                loader_option = st.selectbox("Choose a Loader", ["PyPDFLoader", "PyMuPDFLoader"])

                # Process uploaded files with PyPDFLoader and split_pages=True
                if loader_option == "PyPDFLoader":
                    _,pages=process_document(file, loader_type="PyPDFLoader", split_pages=False)

                # Process uploaded files with PyMuPDFLoader and split_pages=True
                elif loader_option == "PyMuPDFLoader":
                    _,pages=process_document(file, loader_type="PyMuPDFLoader", split_pages=False)




            elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                # Process uploaded files with Docx2txt and split_pages=True
                loader_option = st.selectbox("Choose a Loader", ["Docx2txt", "Unstructured"])
                if loader_option == "Docx2txt":
                    _,pages=process_document(file, loader_type="Docx2txt", split_pages=False)

                # Process uploaded files with Unstructured and split_pages=True
                elif loader_option == "Unstructured":
                    _,pages=process_document(file, loader_type="Unstructured", split_pages=False)



     # Display the content of ZIP files
            elif file_type == "application/zip" or file.name.lower().endswith('.zip'):
                zip_file_names,pages = process_zip(file)
                st.write("Files in the ZIP archive:")
                st.write(zip_file_names)
                st.write("Contents of the first file in the ZIP archive:")
                st.info("Zip file processed")
            else:
                st.warning(f"Unsupported file type: {file.name}")
    except Exception as e:
        st.error(f"Error processing {file.name}: {e}")




    try:
        # Initialize text splitter based on user's choice
        text_splitter_choice = st.selectbox("Choose a text splitter", ["CharacterTextSplitter", "RecursiveCharacterTextSplitter"])
        # chunk_size = st.slider("Chunk Size", min_value=50, max_value=1000, value=100)
        # chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=200, value=20)

        if text_splitter_choice == "RecursiveCharacterTextSplitter":
            #call the function to apply the text splitter
            text=apply_recursive_text_splitter(pages)
            st.info(len(text))
            st.info("text splitter applied")
           

        elif text_splitter_choice == "CharacterTextSplitter":
            # call the function to apply the text splitter
            text=apply_char_text_splitter(pages)
            st.info(len(text))
            st.info("Text splitter applied")

        else:
            st.warning("Invalid text splitter selection")

    except Exception as e:
        st.info("### Error in making chunks ")
        st.error(f"Error processing {file.name}: {e}")    


    try:
        # create a function to apply the embedding model
        embedding_model_choice = st.selectbox("Choose a embedding model", ["BAAI/bge-small-en-v1.5", "intfloat/e5-large-v2"])

        if embedding_model_choice == "BAAI/bge-small-en-v1.5":
            #call the function to apply the embedding model
            embedding=apply_embedding_model1()
            st.info("Embedding model applied")

        elif embedding_model_choice == "intfloat/e5-large-v2":
            # call the function to apply the embedding model
            embedding=apply_embedding_model2()
            st.info("Embedding model applied")
        else:
            st.warning("Invalid embedding model selection")

    except Exception as e:
        st.error(f"Error processing {file.name}:{e}")
    


    try:
        st.info("Storing the embedings in vector database....")
        doc_store=vector_database(text,embedding)
            # Run
        retriever = MultiQueryRetriever(
            retriever=doc_store.as_retriever(), llm_chain=llm_chain, parser_key="lines"
        )  # "lines" is the key (attribute name) of the parsed output

        st.info("Embeddings stored in vector database")


        text_input = st.text_input('Enter Text Below (maximum 800 words):',max_chars=800) 

        submit = st.button('Generate')  

        if submit:

            st.subheader("Output:")

            with st.spinner(text="This may take a moment..."):

                
              unique_docs = retriever.get_relevant_documents(text_input, top_k=2)

              response=user_input(text_input,unique_docs)

            

            st.write(response)
       

    except Exception as e:
        st.error(f"Error processing {file.name}:{e}")

        
















            

            
            




