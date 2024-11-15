from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings


def get_embedding_function(model="bge-large"):
    embeddings = OllamaEmbeddings(model="bge-large")
    return embeddings
