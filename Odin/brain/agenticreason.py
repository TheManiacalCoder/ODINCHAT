import os
import json
import logging
import re
from openai import OpenAI as Client
from .conversation_manager import ConversationManager
from .memory_handler import MemoryHandler

# Set up logging with timestamp and level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class AgenticReasoner:
    def __init__(self, api_key, model_name="gpt-4"):
        """
        Initialize the Agentic Reasoner with an API key and model name.
        """
        self.OPEN_ROUTER_API_KEY = api_key
        self.MODEL_NAME = model_name
        self.client = Client(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.OPEN_ROUTER_API_KEY,
        )
        self.conversation_manager = ConversationManager()
        self.memory_handler = MemoryHandler(self.conversation_manager.memory_dir)
        self.role = None  # Initialize role as None, to be set dynamically
        logging.info("Agentic Reasoner initialized.")

    def is_url(self, text):
        """
        Check if the input text is a URL using a regular expression.
        """
        # Regular expression to detect URLs
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        return bool(url_pattern.search(text))

    def process_query(self, user_message):
        """
        Process the user query and generate a response using the AI model.
        """
        # Check if the user message is a URL
        if self.is_url(user_message):
            # Log that a URL was detected
            logging.info("1 URL detected")

            # Acknowledge the URL submission with a specific response
            response_message = "That's some interesting info!"
            self.conversation_manager.save_conversation(
                f"User: {user_message}\nAI: {response_message}",
                self.conversation_manager.generate_bulleted_summary(response_message),
                self.memory_handler.sentence_to_vec(response_message),
                self.role
            )
            return response_message

        # If it's not a URL, proceed with normal query processing
        # Retrieve previous conversations for context
        previous_conversations = self.conversation_manager.get_previous_conversations()
        if previous_conversations is None:  # Ensure previous_conversations is not None
            previous_conversations = []

        context_messages = [{"role": "system", "content": "You are an AI assistant."}]
        for conv in previous_conversations:
            context_messages.append({"role": "user", "content": conv["user_message"]})
            context_messages.append({"role": "assistant", "content": conv["ai_response"]})

        # Add the current user message
        context_messages.append({"role": "user", "content": user_message})

        # Retrieve the last agentic role from the database
        agentic_role = self.conversation_manager.get_last_agentic_role()
        if agentic_role:
            context_messages.append({"role": "system", "content": f"You are an AI assistant with the role: {agentic_role}."})

        # Generate a response from the AI model
        try:
            completion = self.client.chat.completions.create(
                model=self.MODEL_NAME,
                messages=context_messages,
                extra_headers={"HTTP-Referer": "your_site_url", "X-Title": "your_app_name"}
            )
            if completion.choices and completion.choices[0].message:
                response_message = completion.choices[0].message.content

                # Save the conversation to the database
                self.conversation_manager.save_conversation(
                    f"User: {user_message}\nAI: {response_message}",
                    self.conversation_manager.generate_bulleted_summary(response_message),
                    self.memory_handler.sentence_to_vec(response_message),
                    agentic_role
                )

                return response_message
            else:
                logging.error("Error processing query: No message found in API response.")
                return None
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            return None