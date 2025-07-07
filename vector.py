from typing import List
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

def load_data() -> List[Document]:
    documents = []
    ids = []

    with open("data/merida.txt", "r") as file:
        content = file.readlines()
        content = list(filter(lambda a : a != '\n', content))
        
        for i in range(0, len(content), 3):
            chunk_id = content[i+1].split(': ')[1].rstrip('\n')
            chunk = f"{content[i].rstrip()} {content[i+2].rstrip()}"

            document = Document(
                page_content=chunk,
                id=chunk_id
            )

            documents.append(document)
            ids.append(chunk_id)

    return documents


def create_vector_database(location: str) -> Chroma:
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        collection_name="merida_information",
        persist_directory=location,
        embedding_function=embedding
    )

    return vector_store

load_data()