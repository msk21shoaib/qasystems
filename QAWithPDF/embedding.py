from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.gemini import GeminiEmbedding

from QAWithPDF.data_ingestion import load_data
from QAWithPDF.model_api import load_model

import sys
from exception import customexception
from logger import logging

def download_gemini_embedding(model, document):
    """
    Downloads and initializes a Gemini Embedding model for vector embeddings.

    Returns:
    - QueryEngine: An engine for querying the vector index.
    """
    try:
        logging.info("")
        gemini_embed_model = GeminiEmbedding(model_name="models/embedding-001")

        # Set values directly on Settings
        Settings.llm = model
        Settings.embed_model = gemini_embed_model
        Settings.chunk_size = 512
        Settings.chunk_overlap = 20
        Settings.max_input_size = 4096

        logging.info("")
        index = VectorStoreIndex.from_documents(document, service_context=Settings)
        index.storage_context.persist()

        logging.info("")
        query_engine = index.as_query_engine()
        return query_engine

    except Exception as e:
        raise customexception(e, sys)
