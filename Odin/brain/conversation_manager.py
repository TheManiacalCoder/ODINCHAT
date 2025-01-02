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

# Set up logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

class ConversationManager:
    def __init__(self, file_picker_button=None):
        """
        Initialize the ConversationManager with a memory directory, database, and Word2Vec model.
        """
        self.memory_dir = os.path.join(os.path.dirname(__file__), "Memory")
        os.makedirs(self.memory_dir, exist_ok=True)
        self.conv_folder = None
        self.db_path = None
        self.MODEL_NAME = self._load_model_name()  # Load the model name from config.json
        self.OPEN_ROUTER_API_KEY = None
        self.client = None
        self.memory_handler = MemoryHandler(self.memory_dir)
        self.file_picker = FilePicker(self, file_picker_button)
        self.chatbot_ui = None
        self.file_chunks = []
        self.file_ids = {}
        self.lock = threading.Lock()
        self.embedding_cache = {}  # Cache for embeddings to avoid redundant computations
        self.init_conversation()  # Initialize the conversation and database path

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
                user_message TEXT,
                ai_response TEXT,
                message_summary TEXT,
                embedding TEXT,
                file_name TEXT
            )''')
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
            self.client = Client(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.OPEN_ROUTER_API_KEY,
            )
            logging.info("OpenAI client updated with new API key and model name.")

    def batch_process_queries(self, chunks, file_name=None):
        """
        Process multiple chunks of text in batch mode and train the Word2Vec model on all chunks at once.
        """
        # Ensure the database path is initialized
        if self.db_path is None:
            logging.error("Database path is not initialized. Initializing a new conversation.")
            self.init_conversation()

        # Save all chunks to the database in a single transaction
        self.save_conversations_in_batch(chunks, file_name)

        # Train the Word2Vec model on all chunks in a single batch
        self.memory_handler.train_word2vec(chunks)

    def save_conversations_in_batch(self, chunks, file_name=None):
        """
        Save all chunks to the database in a single batch transaction.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            for chunk in chunks:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                combined_message = f"User: {chunk}\nAI: {chunk}"  # Placeholder for AI response
                summary = self.generate_summary(chunk)
                embedding = self.memory_handler.sentence_to_vec(summary)
                if isinstance(embedding, np.ndarray):
                    embedding = json.dumps(embedding.tolist())

                cursor.execute('''INSERT INTO conversations (timestamp, user_message, ai_response, message_summary, embedding, file_name)
                                  VALUES (?, ?, ?, ?, ?, ?)''',
                               (timestamp, chunk, chunk, summary, embedding, file_name))
            conn.commit()
            logging.info(f"Saved {len(chunks)} chunks to the database in a single transaction.")
        except Exception as e:
            logging.error(f"Error saving chunks to the database: {str(e)}")
        finally:
            conn.close()

    def get_previous_conversations(self, file_name=None):
        """
        Retrieve previous conversations from the database, optionally filtered by file_name.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            if file_name:
                cursor.execute("SELECT user_message, ai_response FROM conversations WHERE file_name = ? ORDER BY timestamp ASC", (file_name,))
            else:
                cursor.execute("SELECT user_message, ai_response FROM conversations ORDER BY timestamp ASC")
            rows = cursor.fetchall()
            conversations = [{"user_message": row[0], "ai_response": row[1]} for row in rows]
            conn.close()
            return conversations
        except Exception as e:
            logging.error(f"Error retrieving previous conversations: {str(e)}")
            return []

    def generate_summary(self, text):
        """
        Generate a concise summary of the text.
        """
        # Example implementation: Take the first 3 sentences
        sentences = text.split('. ')
        summary = "\n".join([f"- {sentence.strip()}" for sentence in sentences[:3]])
        return summary

    def save_conversation(self, user_message, ai_response, summary=None, embedding=None, file_name=None):
        """
        Save the conversation to the database with all fields in a single row.
        """
        with self.lock:  # Use the lock for thread safety
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if summary is None:
                summary = self.generate_summary(ai_response)
            
            if embedding is None:
                embedding = self.memory_handler.sentence_to_vec(summary)
            
            if isinstance(embedding, np.ndarray):
                embedding = json.dumps(embedding.tolist())
            
            cursor.execute('''INSERT INTO conversations (timestamp, user_message, ai_response, message_summary, embedding, file_name)
                              VALUES (?, ?, ?, ?, ?, ?)''',
                           (timestamp, user_message, ai_response, summary, embedding, file_name))
            conn.commit()
            conn.close()
            logging.info("Conversation saved to the database.")

    def process_query(self, user_message):
        """
        Process the user query and generate a response using the AI model.
        """
        try:
            # Retrieve previous conversations for context
            previous_conversations = self.get_previous_conversations()

            # Generate embedding for the user message
            user_message_embedding = self.memory_handler.sentence_to_vec(user_message)

            # Calculate similarity scores for each previous conversation
            similarity_scores = []
            for conv in previous_conversations:
                conv_embedding = self.get_conversation_embedding(conv)
                if conv_embedding is not None:
                    similarity = cosine_similarity([user_message_embedding], [conv_embedding])[0][0]
                    similarity_scores.append((similarity, conv))

            # Sort conversations by similarity score and select the top N
            similarity_scores.sort(reverse=True, key=lambda x: x[0])
            top_n_conversations = [conv for _, conv in similarity_scores[:5]]  # Adjust N as needed

            # Build context messages with the top N conversations
            context_messages = [{"role": "system", "content": "You are an AI assistant. Respond naturally and conversationally."}]
            for conv in top_n_conversations:
                context_messages.append({"role": "user", "content": conv["user_message"]})
                context_messages.append({"role": "assistant", "content": conv["ai_response"]})

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

    def get_conversation_embedding(self, conversation):
        """
        Retrieve or compute the embedding for a conversation.
        """
        if conversation["user_message"] in self.embedding_cache:
            return self.embedding_cache[conversation["user_message"]]

        # Compute the embedding if not cached
        combined_message = f"User: {conversation['user_message']}\nAI: {conversation['ai_response']}"
        embedding = self.memory_handler.sentence_to_vec(combined_message)
        self.embedding_cache[conversation["user_message"]] = embedding
        return embedding