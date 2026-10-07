"""Create the configured Gemini embedding adapter when credentials are available."""

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from document_rag.config import AppConfig, load_config


def create_embeddings(config: AppConfig | None = None) -> GoogleGenerativeAIEmbeddings:
    """Use the Gemini Developer API for document and query embeddings."""
    settings = config or load_config()
    settings.require_google_api_key()
    return GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        api_key=settings.google_api_key,
        output_dimensionality=settings.embedding_dimensions,
        vertexai=False,
    )
