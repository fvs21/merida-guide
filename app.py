import gradio as gr
from dotenv import load_dotenv
import os

import vector, retrieval

load_dotenv()

HUGGING_FACE_TOKEN = os.environ.get("HUGGING_FACE_TOKEN")
DB_LOCATION = "./chroma_db"
ADD_DOCUMENTS = not os.path.exists(DB_LOCATION)
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

def initialize_llm():
    vector_db = vector.create_vector_database(DB_LOCATION)

    if ADD_DOCUMENTS:
        documents = vector.load_data()
        vector_db.add_documents(documents)

    global qa_chain
    qa_chain = retrieval.initialize_llm_qa_chain(
        model=MODEL_NAME,
        temperature=0.7,
        max_tokens=250,
        top_k=3,
        vector_db=vector_db,
        api_token=HUGGING_FACE_TOKEN
    )

def respond(
    message,
    history: list[tuple[str, str]]
):
    answer = retrieval.invoke_qa_chain(
        qa_chain,
        message,
        history
    )

    return answer


def gradio_ui():
    with gr.Blocks() as demo:
        gr.ChatInterface(fn=respond, type="messages")

    demo.launch()

if __name__ == "__main__":
    initialize_llm()
    gradio_ui()