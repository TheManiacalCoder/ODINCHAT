import os
import datetime
import csv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from openai import OpenAI as Client

class ConversationEventHandler(FileSystemEventHandler):
    def __init__(self, manager):
        self.manager = manager

    def on_modified(self, event):
        if event.src_path == self.manager.conversation_csv_path:
            self.manager.process_new_messages()

class ConversationManager:
    def __init__(self):
        self.memory_dir = os.path.join(os.path.dirname(__file__), "Memory")
        os.makedirs(self.memory_dir, exist_ok=True)
        self.conv_folder = None
        self.conversation_csv_path = None
        self.context = []
        self.unique_entries = set()
        self.MODEL_NAME = "gpt-4"
        self.OPEN_ROUTER_API_KEY = None
        self.client = None
        self.observer = None
        self.last_modified_time = None

        self.init_conversation()

    def init_conversation(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conv_folder = os.path.join(self.memory_dir, f"memory_{timestamp}")
        os.makedirs(self.conv_folder, exist_ok=True)
        self.conversation_csv_path = os.path.join(self.conv_folder, "conversations.csv")
        
        if not os.path.exists(self.conversation_csv_path):
            with open(self.conversation_csv_path, "w", encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Message", "Summary"])
        self.load_conversation_from_csv()
        self.start_watching_file()

    def load_conversation_from_csv(self):
        current_modified_time = os.path.getmtime(self.conversation_csv_path)
        if self.last_modified_time is None or self.last_modified_time < current_modified_time:
            self.last_modified_time = current_modified_time
            self.context = []
            self.unique_entries = set()
            if os.path.exists(self.conversation_csv_path):
                with open(self.conversation_csv_path, "r", encoding='utf-8', newline='') as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header row
                    for row in reader:
                        timestamp = row[0]
                        message = row[1]
                        summary = row[2]
                        entry_key = (timestamp, message)
                        if entry_key not in self.unique_entries:
                            self.unique_entries.add(entry_key)
                            self.context.append((timestamp, message, summary))

    def append_to_conversation(self, user_query, ai_response=None, timestamp=None):
        if not timestamp:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if ai_response is not None:
            combined_message = f"User: {user_query}\nAI: {ai_response}"
        else:
            combined_message = f"User: {user_query}"
        
        summary = self.summarize(combined_message)
        
        entry_key = (timestamp, combined_message)
        if entry_key not in self.unique_entries:
            self.unique_entries.add(entry_key)
            self.context.append((timestamp, combined_message, summary))
            self.save_conversation_to_csv(combined_message, summary, timestamp)

    def summarize(self, message):
        if len(message.split()) <= 50:
            return message
        
        summary_prompt = f"Summarize the following message in 50 words or less:\n\n{message}"
        
        try:
            completion = self.client.chat.completions.create(
                model=self.MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an AI assistant tasked with summarizing messages."},
                    {"role": "user", "content": summary_prompt}
                ],
                extra_headers={
                    "HTTP-Referer": "your_site_url",  # Replace with your site URL
                    "X-Title": "your_app_name",       # Replace with your app name
                }
            )
            if completion.choices and completion.choices[0].message:
                return completion.choices[0].message.content.strip()
            else:
                print("Error summarizing message: No message found in API response.")
                return message
        except Exception as e:
            print(f"Error summarizing message: {str(e)}")
            return message

    def process_query(self, user_message):
        self.load_conversation_from_csv()

        self.append_to_conversation(user_message)

        conversation_history = [
            {"role": "system", "content": "You are an AI assistant. I will remember our conversation and provide relevant responses based on previous interactions."}
        ]

        for timestamp, message, summary in self.context:
            if "\n" in message:
                user_part, ai_part = message.split("\n", 1)
                user_part = user_part.replace("User: ", "").strip()
                ai_part = ai_part.replace("AI: ", "").strip()
                conversation_history.append({"role": "user", "content": user_part})
                conversation_history.append({"role": "assistant", "content": ai_part})
            else:
                user_part = message.replace("User: ", "").strip()
                conversation_history.append({"role": "user", "content": user_part})

        try:
            completion = self.client.chat.completions.create(
                model=self.MODEL_NAME,
                messages=conversation_history,
                extra_headers={
                    "HTTP-Referer": "your_site_url",  # Replace with your site URL
                    "X-Title": "your_app_name",       # Replace with your app name
                }
            )
            if completion.choices and completion.choices[0].message:
                response_message = completion.choices[0].message.content
                self.append_to_conversation(user_message, response_message)
                return response_message
            else:
                print("Error processing query: No message found in API response.")
                return None
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return None

    def process_new_messages(self):
        self.load_conversation_from_csv()
        self.cleanup_csv()

    def cleanup_csv(self):
        # Create a temporary file to store valid rows
        temp_csv_path = self.conversation_csv_path + ".tmp"
        with open(self.conversation_csv_path, "r", encoding='utf-8', newline='') as f_in, open(temp_csv_path, "w", encoding='utf-8', newline='') as f_out:
            reader = csv.reader(f_in)
            writer = csv.writer(f_out)
            header = next(reader)
            writer.writerow(header)  # Write the header to the temp file
            
            for row in reader:
                timestamp = row[0]
                message = row[1]
                summary = row[2]
                if "\n" in message:
                    user_part, ai_part = message.split("\n", 1)
                    if user_part.strip().startswith("User: ") and ai_part.strip().startswith("AI: "):
                        writer.writerow(row)
                else:
                    # If the row does not contain a newline, it might be a user message without AI response
                    if message.strip().startswith("User: "):
                        pass  # Skip this row as it's incomplete

        # Replace the original file with the temp file
        os.replace(temp_csv_path, self.conversation_csv_path)
        self.load_conversation_from_csv()

    def clear_conversation(self):
        if self.conv_folder:
            for root, dirs, files in os.walk(self.conv_folder, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.conv_folder)
        
        self.init_conversation()
        self.context = []
        self.unique_entries = set()

    def set_model_name(self, model_name):
        self.MODEL_NAME = model_name
        self.update_client()

    def set_openrouter_api_key(self, api_key):
        self.OPEN_ROUTER_API_KEY = api_key
        self.update_client()

    def update_client(self):
        if self.MODEL_NAME and self.OPEN_ROUTER_API_KEY:
            self.client = Client(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.OPEN_ROUTER_API_KEY,
            )

    def start_watching_file(self):
        if self.conversation_csv_path:
            event_handler = ConversationEventHandler(self)
            self.observer = Observer()
            self.observer.schedule(event_handler, os.path.dirname(self.conversation_csv_path), recursive=False)
            self.observer.start()

    def stop_watching_file(self):
        if self.observer:
            self.observer.stop()
            self.observer.join()

    def save_conversation_to_csv(self, combined_message, summary, timestamp):
        with open(self.conversation_csv_path, "a", encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, combined_message, summary])

# Define the conversation manager instance
conversation_manager = ConversationManager()
conversation_manager.set_model_name("gpt-4")  # Set your desired model name
conversation_manager.set_openrouter_api_key("your_openrouter_api_key")  # Set your OpenRouter API key
conversation_manager.init_conversation()

# Example usage:
user_query = "Tell me a story about a brave knight who saved a kingdom from a dragon."
response = conversation_manager.process_query(user_query)
print(f"AI Response: {response}")