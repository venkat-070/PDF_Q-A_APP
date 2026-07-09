from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import chromadb
from langchain_groq import ChatGroq
import os 
import streamlit as st

api_key = st.secrets["GROQ"]
llm = ChatGroq(
    model= "llama-3.3-70b-versatile",
    api_key=api_key,
    temperature=0.3)
client = chromadb.EphemeralClient()

def process_pdf(pdf_path,name):
    
    existing = [c.name for c in client.list_collections()]
    if name in existing:
        collection = client.get_or_create_collection(name=name)
        return collection.count()
    # Load, chunk, embed, store in ChromaDB
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=300)
    loader = PyPDFLoader(pdf_path)

    pages = loader.load()
    chunks = splitter.split_documents(pages)
    
    collection = client.get_or_create_collection(name=name)
    collection.add(
        documents=[chunk.page_content for chunk in chunks],
        ids = [f"chunk_{i+1}" for i in range(len(chunks))],
        metadatas=[{"Page_no":chunk.metadata['page'],'source':chunk.metadata['source']} for chunk in chunks]
    )
    return len(chunks)

    # Returns: number of chunks stored

def ask_question(question,collection_name):
    collection = client.get_or_create_collection(collection_name)
    results = collection.query(
        query_texts=[question],
        n_results=3
    )
    prompt = prompt = """Use the following context to answer the question accurately. 
    Use reasoning and inference when the answer is implied but not explicitly stated.
    Only say 'I don't know' if there is absolutely no relevant information in the context.

    Context:
    """
    prompt += "".join(results['documents'][0])
    prompt += f"\nquestion: {question}"
    metadata = results["metadatas"][0]
    page_no = [m['Page_no'] for m in metadata]

    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"Chunk {i+1} (Page {meta['Page_no']}):")
        print(doc[:200])
        print("---")

    output = llm.invoke(prompt)
    return output , page_no
