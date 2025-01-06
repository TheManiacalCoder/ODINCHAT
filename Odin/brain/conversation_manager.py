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
        self.client = None
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
                source_type TEXT,  -- e.g., "query", "SCRAPED_INFO", "file"
                url TEXT,          -- Store the URL for scraped content
                file_name TEXT     -- Store the file name for file content
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

    def is_url(self, text):
        """
        Check if the input text is a URL using a regular expression.
        """
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        return bool(url_pattern.search(text))

    def save_to_db(self, content, embedding, source_type, url=None, file_name=None):
        """
        Save content and its embedding to the database.
        If the content is a URL, set the source_type to "SCRAPED_INFO".
        If the content is a file, set the source_type to "file".
        """
        # Check if the content is a URL
        if self.is_url(content):
            source_type = "SCRAPED_INFO"

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute('''
            INSERT INTO embeddings (timestamp, content, embedding, source_type, url, file_name)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (timestamp, content, json.dumps(embedding.tolist()), source_type, url, file_name))

        conn.commit()
        conn.close()
        logging.info(f"Saved {source_type} to the database: {content[:50]}...")  # Log the first 50 chars of content

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

            # Save the chunk to the database
            cursor.execute('''
                INSERT INTO embeddings (timestamp, content, embedding, source_type, file_name)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, chunk, json.dumps(embedding.tolist()), "file", file_name))

        conn.commit()
        conn.close()
        logging.info(f"File chunks saved to the database for file: {file_name}")

    def find_most_relevant_content(self, query_embedding):
        """
        Find the most relevant content from the database using cosine similarity.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Retrieve all embeddings from the database
        cursor.execute("SELECT content, embedding, source_type, url, file_name FROM embeddings")
        rows = cursor.fetchall()

        if not rows:
            logging.warning("No content found in the database.")
            return None

        # Calculate cosine similarity between the query and each embedding
        similarities = []
        for row in rows:
            content, embedding_json, source_type, url, file_name = row
            embedding = np.array(json.loads(embedding_json))
            similarity = cosine_similarity([query_embedding], [embedding])[0][0]
            similarities.append((content, similarity, source_type, url, file_name))

        # Find the most relevant content
        most_relevant_content = max(similarities, key=lambda x: x[1])[0]
        return most_relevant_content

    def generate_response(self, user_message, context):
        """
        Generate a response from the AI model using the provided context.
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an AI assistant. Respond naturally and conversationally."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_message}"}
                ],
                extra_headers={"HTTP-Referer": "your_site_url", "X-Title": "your_app_name"}
            )
            if completion.choices and completion.choices[0].message:
                response_message = completion.choices[0].message.content
                return response_message
            else:
                logging.error("Error generating response: No message found in API response.")
                return None
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return None

    def process_query(self, user_message):
        """
        Process a user query or URL submission.
        If the input is a URL, delegate scraping to the WebAgent, save the content in Markdown format, and return a response.
        If the input is a query, check if it refers to a previously saved article or file and generate a response.
        """
        if self.is_url(user_message):
            # If the input is a URL, delegate scraping to the WebAgent
            logging.info("URL detected, delegating to WebAgent...")
            scraped_content = self.web_agent.scrape_website_content(user_message)
            if scraped_content:
                # Convert scraped content to Markdown format
                markdown_content = self.convert_to_markdown(scraped_content)

                # Save the scraped content to the database with the URL
                self.save_to_db(
                    markdown_content,
                    self.memory_handler.sentence_to_vec(markdown_content),
                    "SCRAPED_INFO",
                    url=user_message  # Save the URL for recall
                )
                logging.info("Website content scraped and saved to database in Markdown format.")

                # Return a response acknowledging the URL submission
                return "The website content has been scraped and saved to memory. How can I assist you further?"
            else:
                logging.error("Failed to scrape the website content.")
                return "Failed to scrape the website content. Please check the URL and try again."
        else:
            # If the input is a query, check if it refers to a previously saved article or file
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check for the most recent scraped content or file content
            cursor.execute("SELECT content FROM embeddings WHERE source_type IN ('SCRAPED_INFO', 'file') ORDER BY timestamp DESC LIMIT 1")
            row = cursor.fetchone()
            conn.close()

            if row:
                # Retrieve the most recent content
                content = row[0]

                # Generate a response using the AI model
                response = self.generate_response(user_message, content)
                return response
            else:
                return "No content has been saved yet. Please provide a URL or upload a file."

    def convert_to_markdown(self, scraped_content):
        """
        Convert scraped content to Markdown format.
        """
        markdown = f"# {scraped_content.get('title', 'No Title')}\n\n"
        markdown += f"## Headings\n"
        for heading_level, headings in scraped_content.get('headings', {}).items():
            markdown += f"### {heading_level.upper()}\n"
            for heading in headings:
                markdown += f"- {heading}\n"
        markdown += "\n## Paragraphs\n"
        for paragraph in scraped_content.get('paragraphs', []):
            markdown += f"{paragraph}\n\n"
        return markdown

    # Rest of the ConversationManager class remains unchanged...
# END OF FILE: C:\Users\Sean Craig\Desktop\AI Python Tools\Odin\brain\conversation_manager.py