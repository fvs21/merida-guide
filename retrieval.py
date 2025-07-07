from typing import List, Tuple
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain, BaseConversationalRetrievalChain

PROMPT_TEMPLATE = '''
You are a tourist guide/assistant for the city of Mérida, Yucatán, México. Use the following pieces of context to answer the question at the end.
Keep the answer concise.

Question: {question}
Context: {context}
'''

def initialize_llm(model: str, temperature: int, max_tokens: int, top_k: int, vector_db: Chroma, api_token: str) -> BaseConversationalRetrievalChain:
    llm = HuggingFaceEndpoint(
        endpoint_url=model,
        max_new_tokens=max_tokens,
        top_k=top_k,
        temperature=temperature,
        task="text-generation",
        huggingfacehub_api_token=api_token
    )

    retriever = vector_db.as_retriever()

    rag_prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["question", "context"]
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        chain_type="stuff",
        verbose=False,
        combine_docs_chain_kwargs={"prompt", rag_prompt}
    )

    return qa_chain

def format_chat_history(history) -> List[str]:
    formatted = []

    for message in history:
        formatted.append("User: ", message[0])
        formatted.append("Assistant: ", message[1])

    return formatted

def invoke_qa_chain(qa_chain: BaseConversationalRetrievalChain, message: str, chat_history: List[Tuple[str, str]]) -> str:
    formatted_chat_history = format_chat_history(chat_history)

    response = qa_chain.invoke(
        {"question": message, "chat_history": formatted_chat_history}
    )

    answer = response["answer"]

    return answer