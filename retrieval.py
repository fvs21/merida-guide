from typing import List, Tuple
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.base import Runnable

PROMPT_TEMPLATE = '''
You are a tourist guide/assistant for the city of Mérida, Yucatán, México. Use the following pieces of context to answer the question at the end.
Keep the answer concise.

Question: {question}
Context: {context}
'''

def initialize_llm_qa_chain(model: str, temperature: int, max_tokens: int, top_k: int, vector_db: Chroma, api_token: str) -> Runnable:
    llm = HuggingFaceEndpoint(
        repo_id=model,
        task="text-generation",
        max_new_tokens = max_tokens,
        top_k = top_k,
        temperature = temperature,
        huggingfacehub_api_token=api_token
    )

    retriever = vector_db.as_retriever()

    rag_prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["question", "context"]
    )

    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=rag_prompt
    )

    qa_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain,
    )

    return qa_chain

def format_chat_history(history) -> List[str]:
    formatted = []

    for message in history:
        formatted.append("User: ", message[0])
        formatted.append("Assistant: ", message[1])

    return formatted

def invoke_qa_chain(qa_chain: Runnable, message: str, chat_history: List[Tuple[str, str]]) -> str:
    formatted_chat_history = format_chat_history(chat_history)

    response = qa_chain.invoke(
        {"question": message, "chat_history": formatted_chat_history, "input": message}
    )

    answer = response["answer"]

    return answer