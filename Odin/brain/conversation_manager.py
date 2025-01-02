import os
import datetime
import sqlite3
import logging
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI as Client
from .memory_handler import MemoryHandler
from .file_picker import FilePicker
import threading

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

class ConversationManager:
    def __init__(self):
        self.memory_dir = os.path.join(os.path.dirname(__file__), "Memory")
        os.makedirs(self.memory_dir, exist_ok=True)
        self.conv_folder = None
        self.db_path = None
        self.MODEL_NAME = self._load_model_name()
        self.OPEN_ROUTER_API_KEY = None
        self.client = None
        self.memory_handler = MemoryHandler(self.memory_dir)
        self.file_picker = FilePicker(self)
        self.chatbot_ui = None
        self.file_chunks = []
        self.file_ids = {}
        self.lock = threading.Lock()  # Add a lock for thread safety
        self.init_conversation()

    def _load_model_name(self):
        """
        Load the MODEL_NAME from the config.json file.
        """
        config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
        try:
            with open(config_path, "r", encoding="utf-8") as config_file:
                config = json.load(config_file)
                return config["MODEL_NAME"]
        except Exception as e:
            logging.error(f"Failed to load MODEL_NAME from config.json: {str(e)}")
            raise ValueError("MODEL_NAME must be defined in config.json")

    def set_openrouter_api_key(self, api_key):
        """
        Set the OpenRouter API key and update the client.
        """
        self.OPEN_ROUTER_API_KEY = api_key
        self.update_client()

    def set_model_name(self, model_name):
        """
        Set the model name and update the client.
        """
        self.MODEL_NAME = model_name
        self.update_client()

    def update_client(self):
        """
        Update the OpenAI client with the current API key and model name.
        """
        if self.MODEL_NAME and self.OPEN_ROUTER_API_KEY:
            self.client = Client(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.OPEN_ROUTER_API_KEY,
            )
            logging.info("OpenAI client updated with new API key and model name.")

    def init_conversation(self):
        """
        Initialize a new conversation with a new folder, database, and Word2Vec model.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conv_folder = os.path.join(self.memory_dir, f"memory_{timestamp}")
        os.makedirs(self.conv_folder, exist_ok=True)
        self.db_path = os.path.join(self.conv_folder, "conversations.db")
        self.memory_handler = MemoryHandler(self.conv_folder)
        self.init_db()
        if not os.path.exists(self.memory_handler.model_path):
            logging.info("Initializing Word2Vec model with default data.")
            self.memory_handler.load_or_train_word2vec_model()

    def init_db(self):
        """
        Initialize the SQLite database for the conversation.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                message TEXT,
                message_summary TEXT,
                embedding TEXT
            )''')
        conn.commit()
        conn.close()
        logging.info(f"Database initialized at {self.db_path}")

    def get_previous_conversations(self):
        """
        Retrieve previous conversations from the database, but only the summaries.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT message_summary FROM conversations ORDER BY timestamp ASC")
            rows = cursor.fetchall()
            conversations = [{"message": row[0]} for row in rows]  # Use summaries only
            conn.close()
            return conversations
        except Exception as e:
            logging.error(f"Error retrieving previous conversations: {str(e)}")
            return []

    def process_query(self, user_message):
        try:
            # Retrieve previous conversations for context (summaries only)
            previous_conversations = self.get_previous_conversations()
            context_messages = [{"role": "system", "content": "You are an AI assistant."}]
            for conv in previous_conversations:
                context_messages.append({"role": "user", "content": conv["message"]})

            # Add the current user message
            context_messages.append({"role": "user", "content": user_message})

            # Generate a response from the AI model
            completion = self.client.chat.completions.create(
                model=self.MODEL_NAME,
                messages=context_messages,
                extra_headers={"HTTP-Referer": "your_site_url", "X-Title": "your_app_name"}
            )
            if completion.choices and completion.choices[0].message:
                response_message = completion.choices[0].message.content
                self.save_conversation(user_message, response_message)
                return response_message
            else:
                logging.error("Error processing query: No message found in API response.")
                return None
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            return None

    def save_conversation(self, user_message, ai_response, summary=None, embedding=None):
        """
        Save the conversation to the database with all fields in a single row.
        Also, retrain the Word2Vec model on each summary chunk separately.
        """
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            combined_message = f"User: {user_message}\nAI: {ai_response}"
            
            if summary is None:
                summary = self.generate_summary(ai_response)
            
            if embedding is None:
                embedding = self.memory_handler.sentence_to_vec(summary)
            
            if isinstance(embedding, np.ndarray):
                embedding = json.dumps(embedding.tolist())
            
            cursor.execute('''INSERT INTO conversations (timestamp, message, message_summary, embedding)
                              VALUES (?, ?, ?, ?)''',
                           (timestamp, combined_message, summary, embedding))
            conn.commit()
            conn.close()
            logging.info("Conversation saved to the database.")

            summary_chunks = self.split_summary_into_chunks(summary)
            self.memory_handler.train_word2vec(summary_chunks)

    def generate_summary(self, ai_response):
        """
        Generate a concise bulleted list of the main points from the AI response.
        """
        # Example implementation: Split the response into sentences and take the first few
        sentences = ai_response.split('. ')
        summary = "\n".join([f"- {sentence.strip()}" for sentence in sentences[:3]])  # Take first 3 sentences
        return summary

    def split_summary_into_chunks(self, summary):
        """
        Split the summary into chunks (e.g., bullet points or sentences).
        """
        # Split by newline for bullet points or by period for sentences
        chunks = summary.split('\n')  # Split by newline for bullet points
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]  # Remove empty chunks
        return chunks

    def notify(self, message):
        """
        Notify the user by logging the message and displaying it in the chatbot UI if available.
        """
        logging.info(message)
        if self.chatbot_ui and hasattr(self.chatbot_ui, 'display_response'):
            self.chatbot_ui.display_response({"type": "text", "content": message})
        else:
            logging.warning("Chatbot UI is not initialized. Cannot display response.")

    def pick_file(self):
        """
        Delegate file picking to the FilePicker instance.
        """
        if hasattr(self, 'file_picker'):
            self.file_picker.pick_file()
        else:
            logging.error("FilePicker is not initialized.")

    def clear_conversation(self, new_conversation=True):
        """
        Clear the current conversation and optionally start a new one.
        """
        if self.conv_folder:
            for root, dirs, files in os.walk(self.conv_folder, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.conv_folder)
        
        if new_conversation:
            self.init_conversation()