# Meeting Memory Assistant - Project Proposal

## 1. Overview

This project develops a GenAI powered Meeting Memory Assistant that enables users to query and retrieve information from their accumulated meeting notes using natural language. Built on a Retrieval Augmented Generation (RAG) architecture, the system ingests PDF meeting notes, stores them in a local vector database, and provides accurate, context aware answers to user questions. Our primary objective is to eliminate the time consuming task of manually searching through meeting documentation, allowing professionals to instantly recall decisions, action items, attendees, and discussions from any past meeting.

## 2. Problem Statement

Employees attend numerous meetings weekly, generating substantial documentation that quickly becomes difficult to navigate. Critical information, such as who was assigned a task, what decisions were made, or what was discussed about a specific topic—often gets buried in lengthy notes across multiple files. Manually searching through these documents is inefficient and error prone, leading to missed follow ups, duplicated discussions, and lost institutional knowledge. This project addresses this gap by providing an intelligent, conversational interface that makes meeting knowledge instantly accessible and queryable.

## 3. Proposed Solution

The proposed solution employs a RAG architecture to combine the strengths of semantic search with large language model generation. The system operates in two phases:

**Ingestion Phase:**
- PDF meeting notes are loaded and parsed from a local directory
- Documents are split into semantically meaningful chunks (800 characters with 150-character overlap)
- Chunks are embedded using Amazon Titan Embed Text v2 and stored in a Chroma vector database with local persistence

**Query Phase:**
- User questions are converted to embeddings and matched against stored chunks using similarity search
- The top 4 most relevant chunks are retrieved as context
- Amazon Nova Lite LLM generates a precise answer grounded in the retrieved meeting content
- Source attribution is provided to trace answers back to specific meeting documents

The CLI interface enables rapid, interactive querying while maintaining simplicity and low resource overhead.

## 4. Technology & Tools

| Category | Technology | Purpose |
|----------|------------|---------|
| **Orchestration** | LangChain >= 0.3.0 | RAG chain construction and prompt management |
| **LLM** | Amazon Nova Lite (amazon.nova-lite-v1:0) | Response generation via AWS Bedrock |
| **Embeddings** | Amazon Titan Embed Text v2 (amazon.titan-embed-text-v2:0) | Document and query vectorization |
| **Vector Store** | Chroma (langchain-chroma) | Local vector storage with persistence |
| **Document Processing** | PyPDF | PDF parsing and text extraction |
| **Cloud Platform** | AWS Bedrock (us-east-1) | Managed LLM and embedding inference |
| **Configuration** | python-dotenv | Environment variable management |
| **Language** | Python 3.10+ | Core implementation |
