import os
import datetime
import numpy as np
import csv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from openai import OpenAI as Client
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import sqlite3
from .memory_handler import MemoryHandler
from .conversation_manager import ConversationManager

conversation_manager = ConversationManager()

def process_response_with_word2vec(self, response):
    """
    Process the response with Word2Vec for memory and embeddings.
    Returns the embedding vector.
    """
    try:
        # Convert the response to a vector using Word2Vec
        embedding = self.memory_handler.sentence_to_vec(response)
        if embedding:
            logging.info("Response processed with Word2Vec and embedded in memory.")
            return embedding
        else:
            logging.warning("Failed to generate Word2Vec embedding for the response.")
            return None
    except Exception as e:
        logging.error(f"Error processing response with Word2Vec: {str(e)}")
        return None