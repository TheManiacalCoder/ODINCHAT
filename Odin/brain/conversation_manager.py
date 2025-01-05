import os
import sqlite3
import logging
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from .memory_handler import MemoryHandler
from .web_agent import WebAgent
import datetime
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class ConversationManager:
    def __init__(self):
        """
        Initialize the ConversationManager.
        """
        self.memory_dir = os.path.join(os.path.dirname(__file__), "Memory")
        os.makedirs(self.memory_dir, exist_ok=True)
        self.conv_folder = self.create_conversation_folder()
        self.db_path = os.path.join(self.conv_folder, "conversations.db")
        self.MODEL_NAME = None
        self.OPEN_ROUTER_API_KEY = None
        self.client = None
        self.memory_handler = MemoryHandler(self.conv_folder)
        self.web_agent = WebAgent(self)
        self.init_db()

    def create_conversation_folder(self):
        """
        Create a new folder for the current conversation.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        conv_folder = os.path.join(self.memory_dir, f"conversation_{timestamp}")
        os.makedirs(conv_folder, exist_ok=True)
        logging.info(f"Created new conversation folder: {conv_folder}")
        return conv_folder

    def init_db(self):
        """
        Initialize the SQLite database for storing conversations and file/website chunks.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS file_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                chunk TEXT,
                embedding TEXT,
                file_name TEXT
            )''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS website_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                chunk TEXT,
                embedding TEXT,
                url TEXT,
                title TEXT
            )''')
        conn.commit()
        conn.close()
        logging.info(f"Database initialized at {self.db_path}")

    def save_chunks_to_db(self, chunks, file_name):
        """
        Save file chunks to the database with embeddings.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for chunk in chunks:
            embedding = self.memory_handler.sentence_to_vec(chunk)
            cursor.execute('''
                INSERT INTO file_chunks (timestamp, chunk, embedding, file_name)
                VALUES (?, ?, ?, ?)
            ''', (
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                chunk,
                json.dumps(embedding.tolist()) if embedding is not None else None,
                file_name
            ))
        conn.commit()
        conn.close()
        logging.info(f"Saved {len(chunks)} chunks to the database for file: {file_name}")

    def set_openrouter_api_key(self, api_key):
        """
        Set the OpenRouter API key.
        """
        self.OPEN_ROUTER_API_KEY = api_key
        self.update_client()

    def set_model_name(self, model_name):
        """
        Set the AI model name.
        """
        self.MODEL_NAME = model_name
        self.update_client()

    def update_client(self):
        """
        Update the OpenAI client with the current API key and model name.
        """
        if self.MODEL_NAME and self.OPEN_ROUTER_API_KEY:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.OPEN_ROUTER_API_KEY,
            )
            logging.info("OpenRouter client updated with new API key and model name.")

    def process_query(self, user_message):
        """
        Process a user query. If a file or website is available, use it; otherwise, respond naturally.
        """
        try:
            # Check if the user message is a URL
            if self.web_agent.is_url(user_message):
                url = self.web_agent.extract_urls(user_message)[0]
                scraped_data = self.web_agent.scrape_url(url)
                if not scraped_data:
                    return "Failed to scrape the website. Please try again."
                self.web_agent.save_seo_data(url, scraped_data)
                return "Website scraped successfully. Please ask your question about the content."

            # Check if there are file chunks in the database
            file_chunks = self.get_all_chunks()
            if file_chunks:
                # Find the most relevant chunk for the query
                relevant_chunk = self.find_most_relevant_chunk(user_message)
                if relevant_chunk:
                    truncated_chunk = self.truncate_text(relevant_chunk, max_tokens=2000)
                    response = self.process_chunks_with_ai([truncated_chunk], user_message)
                    return response
                else:
                    return "No relevant content found in the uploaded files."

            # If no file or website is available, respond naturally
            return self.process_natural_query(user_message)

        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            return "An error occurred while processing your query."

    def process_natural_query(self, user_message):
        """
        Process a query naturally without relying on files or websites.
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an AI assistant. Respond naturally and conversationally."},
                    {"role": "user", "content": user_message}
                ],
                extra_headers={"HTTP-Referer": "your_site_url", "X-Title": "your_app_name"}
            )
            if completion.choices and completion.choices[0].message:
                return completion.choices[0].message.content
            else:
                logging.error("Error processing query: No message found in API response.")
                return "Sorry, I couldn't generate a response. Please try again."
        except Exception as e:
            logging.error(f"Error processing natural query: {str(e)}")
            return "An error occurred while processing your query."

    def find_most_relevant_chunk(self, query):
        """
        Find the most relevant chunk for a given query using cosine similarity.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT chunk, embedding FROM file_chunks")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return None

        chunks = [row[0] for row in rows]
        embeddings = [np.array(json.loads(row[1])) for row in rows]

        query_embedding = self.memory_handler.sentence_to_vec(query)
        similarities = [cosine_similarity([query_embedding], [embedding])[0][0] for embedding in embeddings]

        most_relevant_index = np.argmax(similarities)
        return chunks[most_relevant_index]

    def get_all_chunks(self):
        """
        Retrieve all file chunks from the database.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT chunk FROM file_chunks")
        rows = cursor.fetchall()
        conn.close()
        return [row[0] for row in rows]

    def truncate_text(self, text, max_tokens=2000):
        """
        Truncate text to ensure it stays within the token limit.
        """
        words = text.split()
        if len(words) > max_tokens:
            truncated_text = " ".join(words[:max_tokens])
            return truncated_text + "..."
        return text

    def process_chunks_with_ai(self, chunks, query):
        """
        Process chunks of text with the AI model to generate a response.
        """
        context = "\n\n".join(chunks)
        try:
            completion = self.client.chat.completions.create(
                model=self.MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an AI assistant. Respond naturally and conversationally."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                ],
                extra_headers={"HTTP-Referer": "your_site_url", "X-Title": "your_app_name"}
            )
            if completion.choices and completion.choices[0].message:
                return completion.choices[0].message.content
            else:
                logging.error("Error processing query: No message found in API response.")
                return None
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            return None