
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter


def apply_char_text_splitter(pages):
    # Apply text splitter
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    return text_splitter.split_documents(pages)

def apply_recursive_text_splitter(pages):
    # Apply text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    return text_splitter.split_documents(pages)