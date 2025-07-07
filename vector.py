from typing import List
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import json

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

    with open("data/places.json", "r") as file:
        json_data = json.loads(file.read())

    for i, place in enumerate(json_data):
        id = int(ids[-1])+1+i

        name = place["name"]
        query = place["query"]

        if isinstance(query, str):
            urls = [query]
        else:
            urls = query

        if len(urls) == 1:
            place_data = f"{name} is located in here {urls[0]}"
        else:
            locations = "\n".join(f"{loc['location']}: {loc['url']} - " for loc in urls)
            place_data = f"{name} has multiple locations, you can find them here: {locations} "
            
        print(place_data)
        document = Document(
            page_content=place_data,
            id=str(id)
        )

        documents.append(document)
        ids.append(str(id))

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