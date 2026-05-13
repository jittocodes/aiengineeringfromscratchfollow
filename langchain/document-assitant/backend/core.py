import os
from typing import Any, Dict, List

from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectorstore = PineconeVectorStore(
    index_name=os.environ['INDEX_NAME'],
    embedding=embeddings
)

model = init_chat_model(
    model = 'gpt-4.1',
    model_provider = "openai"
)

@tool(response_format = "content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain."""
    retrieved_docs = vectorstore.as_retriever().invoke(query, k = 4)
    
    serialilzed = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'unknown')}\n\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    
    return serialilzed, retrieved_docs

def run_llm(query: str) -> Dict[str, Any]:
    """Run the RAG Pipeline to answer a query using the documentation that is retrieved

    Args:
        query (str): The user's questions

    Returns:
        Dict[str, Any]: 
            - answer: The answer generated
            - context: List of the retrieved documents
    """
    
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )
    
    agent = create_agent(
        model= model,
        tools=[retrieve_context],
        system_prompt=system_prompt
    )
    
    messages = [{"role" : "user", "content" : query}]
    
    response = agent.invoke({"messages": messages})
    
    # Since there will be a lot of calls in between
    answer = response["messages"][-1].content
    
    context_docs = []
    
    for message in response["messages"]:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)
    
    return {
        "answer" : answer,
        "context" : context_docs
    }

if __name__ == '__main__':
    result = run_llm(query="what are deep agents?")
    print(result)