"""Configuration for Meeting Memory Assistant."""

import os
import boto3
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings

load_dotenv()

# AWS Region
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MEETING_NOTES_DIR = os.path.join(DATA_DIR, "meeting_notes")
CHROMA_DB_DIR = os.path.join(DATA_DIR, "chroma_db")

# Chunking settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

def get_bedrock_client():
    """Get the AWS Bedrock client."""
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=AWS_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    )

def get_llm():
    """Get the LLM instance."""
    return ChatBedrockConverse(
        client=get_bedrock_client(),
        model_id="amazon.nova-lite-v1:0",
        temperature=0
    )    

def get_embeddings():
    """Get the embeddings instance."""
    return BedrockEmbeddings(
        client=get_bedrock_client(),
        model_id="amazon.titan-embed-text-v2:0",
    )
