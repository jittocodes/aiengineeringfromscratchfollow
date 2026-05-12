from dotenv import load_dotenv

load_dotenv()


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import os
from langchain_ollama import ChatOllama
from operator import itemgetter

print("Initializing components...")

embeddings = OpenAIEmbeddings()
llm = ChatOllama(model='qwen3:1.7b')
vector_store = PineconeVectorStore(
    index_name= os.environ['INDEX_NAME'],
    embedding=embeddings
)


retriever = vector_store.as_retriever(search_kwards={"k" : 5})

prompt_template = ChatPromptTemplate.from_template("""
    Answer the question based only on the following context: 
        {context}
    
    Question: {question}
    
    Provide a detailed answer:
    """
)

def format_docs(docs):
    """Format retrieved documents in a single string"""
    return "\n\n".join(doc.page_content for doc in docs)

print("Retrieving...")
query = "What is pinecone in machine learning?"

# Raw Implementation
# result = llm.invoke([HumanMessage(content=query)])
# print(f"Raw Query: {result.content}")

#RAG Implementation

def retrieval_chain_without_lcel(query: str):
    
    docs = retriever.invoke(query)
    combined_docs = format_docs(docs)
    
    messages = prompt_template.format_messages(
        context = combined_docs,
        question = query
    )
    
    response = llm.invoke(messages)
    return response.content

def retrieval_chain_with_lcel():
    ret_chain = (
        RunnablePassthrough.assign(
            context = itemgetter("question") | 
        retriever | format_docs )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    return ret_chain

chainn = retrieval_chain_with_lcel()
result_with_lcel = chainn.invoke({"question" : query})