"""Meeting Memory Assistant - Query interface."""

import os
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import CHROMA_DB_DIR, get_llm, get_embeddings

# System prompt for meeting assistant
SYSTEM_PROMPT = """You are a Meeting Memory Assistant that helps users recall information from their meeting notes.

GUIDELINES:
1. Answer ONLY based on the provided context from meeting notes
2. If the information is not in the context, say "I couldn't find this in the meeting notes"
3. Cite the source file when possible
4. Be concise and direct

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def format_docs(docs):
    """Format retrieved documents for the prompt."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source_file", "Unknown")
        formatted.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def get_vector_store():
    """Load the existing vector store."""
    if not os.path.exists(CHROMA_DB_DIR):
        return None
    
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name="meeting_notes",
    )


def create_rag_chain(vector_store):
    """Create the RAG chain."""
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )
    
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
    llm = get_llm()
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain


def main():
    """Main query loop."""
    print("=" * 50)
    print("Meeting Memory Assistant")
    print("=" * 50)
    
    # Load vector store
    vector_store = get_vector_store()
    
    if vector_store is None:
        print("\nNo vector store found. Run 'python ingest.py' first.")
        return
    
    # Check if there are documents
    collection = vector_store._collection
    if collection.count() == 0:
        print("\nVector store is empty. Run 'python ingest.py' first.")
        return
    
    print(f"\nLoaded {collection.count()} document chunks")
    print("\nAsk questions about your meeting notes.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    # Create RAG chain
    chain = create_rag_chain(vector_store)
    
    # Query loop
    while True:
        try:
            question = input("You: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            print("\nAssistant: ", end="", flush=True)
            response = chain.invoke(question)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
