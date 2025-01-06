# START OF FILE: C:\Users\Sean Craig\Desktop\AI Python Tools\Odin\brain\conversation_manager.py
import os
import sqlite3
import logging
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from .memory_handler import MemoryHandler
import datetime
from bs4 import BeautifulSoup
import requests
import re
from .web_agent import WebAgent  # Import WebAgent
from gensim.utils import simple_preprocess  # Import simple_preprocess

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class ConversationManager:
    def __init__(self):
        """
        Initialize the ConversationManager with a memory directory and database.
        """
        self.memory_dir = os.path.join(os.path.dirname(__file__), "Memory")
        os.makedirs(self.memory_dir, exist_ok=True)
        self.conv_folder = self.create_conversation_folder()  # Create a new conversation folder
        self.db_path = os.path.join(self.conv_folder, "conversations.db")  # Save .db in the most recent conversation folder
        self.MODEL_NAME = None
        self.OPEN_ROUTER_API_KEY = None
        self.client = None  # Initialize the OpenAI client as None
        self.memory_handler = MemoryHandler(self.conv_folder)  # MemoryHandler uses the conversation folder
        self.web_agent = WebAgent(self)  # Initialize WebAgent
        self.init_db()

    def create_conversation_folder(self):
        """
        Create a new conversation folder with a unique timestamp.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        conv_folder = os.path.join(self.memory_dir, f"conversation_{timestamp}")
        os.makedirs(conv_folder, exist_ok=True)
        logging.info(f"Created new conversation folder: {conv_folder}")
        return conv_folder

    def init_db(self):
        """
        Initialize the SQLite database for storing embeddings and metadata.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create a single table for all embeddings, with a `source_type` column to indicate the type of content
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                content TEXT,
                embedding TEXT,
                source_type TEXT  -- e.g., "user_query", "SCRAPED_INFO", "file"
            )
        ''')

        conn.commit()
        conn.close()
        logging.info(f"Database initialized at {self.db_path}")

    def set_openrouter_api_key(self, api_key):
        """
        Set the OpenRouter API key and update the OpenAI client.
        """
        self.OPEN_ROUTER_API_KEY = api_key
        self.update_client()

    def set_model_name(self, model_name):
        """
        Set the model name and update the OpenAI client.
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
            logging.info("OpenRouter client updated.")
        else:
            logging.warning("OpenRouter client not updated: API key or model name missing.")

    def save_to_db(self, content, embedding, source_type):
        """
        Save content and its embedding to the database.
        - `source_type`: Can be "user_query", "SCRAPED_INFO", or "file".
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute('''
            INSERT INTO embeddings (timestamp, content, embedding, source_type)
            VALUES (?, ?, ?, ?)
        ''', (timestamp, content, json.dumps(embedding.tolist()), source_type))

        conn.commit()
        conn.close()
        logging.info(f"Saved {source_type} to the database: {content[:50]}...")  # Log the first 50 chars of content

        # Retrain Word2Vec model after saving new content
        self.retrain_word2vec_model(content)

    def save_chunks_to_db(self, chunks, file_name):
        """
        Save file chunks and their embeddings to the database.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for chunk in chunks:
            # Generate an embedding for the chunk
            embedding = self.memory_handler.sentence_to_vec(chunk)

            # Save the chunk to the database with source_type as "file"
            cursor.execute('''
                INSERT INTO embeddings (timestamp, content, embedding, source_type)
                VALUES (?, ?, ?, ?)
            ''', (timestamp, chunk, json.dumps(embedding.tolist()), "file"))

        conn.commit()
        conn.close()
        logging.info(f"File chunks saved to the database for file: {file_name}")

    def retrain_word2vec_model(self, new_content):
        """
        Retrain the Word2Vec model with new content.
        """
        try:
            # Tokenize the new content into sentences
            sentences = [simple_preprocess(new_content)]
            # Update the Word2Vec model with the new sentences
            self.memory_handler.update_word2vec_model(sentences)
            logging.info("Word2Vec model retrained with new content.")
        except Exception as e:
            logging.error(f"Error retraining Word2Vec model: {str(e)}")

    def find_most_relevant_content(self, query_embedding, source_type=None):
        """
        Find the most relevant content from the database using cosine similarity.
        - `source_type`: Optional filter to search for specific content types (e.g., "file", "SCRAPED_INFO", "user_query").
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Retrieve all embeddings from the database (optionally filtered by source_type)
        if source_type:
            cursor.execute("SELECT content, embedding FROM embeddings WHERE source_type = ?", (source_type,))
        else:
            cursor.execute("SELECT content, embedding FROM embeddings")

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            logging.warning("No content found in the database.")
            return None

        # Calculate cosine similarity between the query and each embedding
        similarities = []
        for row in rows:
            content, embedding_json = row
            embedding = np.array(json.loads(embedding_json))
            similarity = cosine_similarity([query_embedding], [embedding])[0][0]
            similarities.append((content, similarity))

        # Find the most relevant content
        most_relevant_content = max(similarities, key=lambda x: x[1])[0]
        return most_relevant_content

    def process_query(self, user_message):
        """
        Process a user query or URL submission.
        If the input is a URL, delegate scraping to the WebAgent, save the content, and return a response.
        If the input is a query, generate a response using the AI model.
        """
        if self.is_url(user_message):
            # If the input is a URL, delegate scraping to the WebAgent
            logging.info("URL detected, delegating to WebAgent...")
            scraped_content = self.web_agent.scrape_website_content(user_message)
            if scraped_content:
                # Save the scraped content to the database
                self.save_to_db(
                    json.dumps(scraped_content, ensure_ascii=False),
                    self.memory_handler.sentence_to_vec(json.dumps(scraped_content)),
                    "SCRAPED_INFO"
                )
                logging.info("Website content scraped and saved to database.")
                return "The website content has been scraped and saved to memory. How can I assist you further?"
            else:
                logging.error("Failed to scrape the website content.")
                return "Failed to scrape the website content. Please check the URL and try again."
        else:
            # Perform a vector search to find the most relevant content
            query_embedding = self.memory_handler.sentence_to_vec(user_message)
            most_relevant_content = self.find_most_relevant_content(query_embedding)
            if most_relevant_content:
                # Use the OpenAI API to generate a response based on the most relevant content
                try:
                    completion = self.client.chat.completions.create(
                        model=self.MODEL_NAME,
                        messages=[
                            {"role": "system", "content": "You are an AI assistant. Respond naturally and conversationally."},
                            {"role": "user", "content": f"Here is some relevant context:\n\n{most_relevant_content}\n\nUser: {user_message}"}
                        ],
                        extra_headers={"HTTP-Referer": "your_site_url", "X-Title": "your_app_name"}
                    )
                    if completion.choices and completion.choices[0].message:
                        response_message = completion.choices[0].message.content

                        # Save the conversation to the database
                        self.save_to_db(
                            f"User: {user_message}\nAI: {response_message}",
                            self.memory_handler.sentence_to_vec(response_message),
                            "user_query"
                        )

                        return response_message
                    else:
                        logging.error("Error processing query: No message found in API response.")
                        return None
                except Exception as e:
                    logging.error(f"Error processing query: {str(e)}")
                    return None
            else:
                # If no relevant content is found, generate a response using the AI model
                if not self.client:
                    logging.error("OpenAI client is not initialized. Please set the API key and model name.")
                    return "Error: OpenAI client is not initialized. Please set the API key and model name."

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
                        response_message = completion.choices[0].message.content

                        # Save the conversation to the database
                        self.save_to_db(
                            f"User: {user_message}\nAI: {response_message}",
                            self.memory_handler.sentence_to_vec(response_message),
                            "user_query"
                        )

                        return response_message
                    else:
                        logging.error("Error processing query: No message found in API response.")
                        return None
                except Exception as e:
                    logging.error(f"Error processing query: {str(e)}")
                    return None

    def is_url(self, text):
        """
        Check if the input text is a URL using a regular expression.
        """
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        return bool(url_pattern.search(text))
# END OF FILE: C:\Users\Sean Craig\Desktop\AI Python Tools\Odin\brain\conversation_manager.py