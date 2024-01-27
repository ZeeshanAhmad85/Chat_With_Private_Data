import tempfile
import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
import fitz  # PyMuPDF
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, Docx2txtLoader
import zipfile
import mimetypes





def process_document(uploaded_file, loader_type,split_pages=True):
    file_names = []

    # If the file is within a ZIP archive, use its name and type
    if isinstance(uploaded_file, tuple):
        file_name, file_obj = uploaded_file
        file_type, _ = mimetypes.guess_type(file_name)
    else:
        # If it's a regular file object, use its name and type
        file_name = uploaded_file.name
        file_type, _ = mimetypes.guess_type(file_name)
        file_obj = uploaded_file

    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file_obj.read())
        temp_file_path = temp_file.name

    pages = []  # Initialize an empty list to store pages

    try:
        if file_type == "application/pdf":
            # Choose the appropriate loader based on the loader_type parameter
            if loader_type == "PyPDFLoader":
                loader = PyPDFLoader(temp_file_path)
            elif loader_type == "PyMuPDFLoader":
                loader = PyMuPDFLoader(temp_file_path)
            else:
                raise ValueError("Invalid loader_type for PDF. Supported values are 'PyPDFLoader' and 'PyMuPDFLoader'.")

            if split_pages:
                # Load and split pages
                pages = loader.load_and_split()
            else:
                # Load the entire document as a single entity
                pages = loader.load()

        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # Handling DOCX files
            if loader_type == "Docx2txt":
                loader = Docx2txtLoader(temp_file_path)
            elif loader_type == "Unstructured":
                loader = UnstructuredWordDocumentLoader(temp_file_path)
            else:
                raise ValueError("Invalid loader_type for DOCX. Supported values are 'Docx2txt' and 'Unstructured'.")
            if split_pages:
                # Load and split pages
                pages = loader.load_and_split()
            else:
                # Load the entire document as a single entity
                pages = loader.load()

        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        # Display the content of the first page or entire document as an example
        if pages:
            st.write(f"Content of the document of {file_name} using {loader_type}:")
            st.info("File processed")

    except (fitz.EmptyFileError, Exception) as e:
        st.write(f"Error processing {file_name}: {e}")

    finally:
        # Clean up: delete the temporary file
        if temp_file_path:
            st.write(f"Cleaning up: Deleting temporary file {temp_file_path}")
            os.remove(temp_file_path)

    return file_names,pages  # Return the pages variable







def process_zip(zip_file):
    file_names = []
    pages = []

    # Extract the contents of the ZIP file to a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            file_names = zip_ref.namelist()
            zip_ref.extractall(temp_dir)

        # Process each file in the extracted directory
        for file_name in file_names:
            file_path = os.path.join(temp_dir, file_name)

            # Determine the file type using mimetypes
            file_type, _ = mimetypes.guess_type(file_name)

            # Open the file as a tuple (name, file object)
            with open(file_path, 'rb') as file_obj:
                file_tuple = (file_name, file_obj)

                # Process each file based on its type
                if file_type == "application/pdf":
                    # Process PDF files
                    page=process_document(file_tuple, loader_type="PyPDFLoader", split_pages=True)
                    pages.append(page)
                elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    # Process DOCX files
                    page=process_document(file_tuple, loader_type="Docx2txt", split_pages=True)
                    pages.append(page)
                else:
                    # Handle other file types or display a message
                    st.warning(f"Unsupported file type: {file_name}")

    return file_names, pages  # Return the file names and pages variables