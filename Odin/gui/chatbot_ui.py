# chatbot_ui.py
import customtkinter as ctk
from brain.ai_memory import conversation_manager
import threading
import queue
import logging
from .CustomText import CustomText
from .stream_response import stream_response

# Set up logging
logging.basicConfig(level=logging.DEBUG)

class ChatbotUI:
    def __init__(self, master):
        self.master = master
        self.conversation_manager = conversation_manager
        self.widgets = {
            'text_box': None,
            'scrollbar': None,
            'entry': None,
            'top_frame': None,
            'text_frame': None,
            'entry_frame': None,
        }
        self.response_queue = queue.Queue()
        self.stop_streaming = False
        self.streaming_thread = None
        self.lock = threading.Lock()
        self.create_widgets()

        # Ensure the conversation_manager has a reference to this ChatbotUI instance
        self.conversation_manager.chatbot_ui = self

    def create_widgets(self):
        self.initialize_top_frame()
        self.initialize_text_frame()
        self.initialize_entry_frame()

    def initialize_top_frame(self):
        from .chatbot_buttons import ChatbotButtons
        self.widgets['top_frame'] = ctk.CTkFrame(self.master)
        self.widgets['top_frame'].pack(expand=False, fill=ctk.X, padx=(5, 10), pady=10)
        ChatbotButtons(self.widgets['top_frame'], self)

    def initialize_text_frame(self):
        self.widgets['text_frame'] = ctk.CTkFrame(self.master)
        self.widgets['text_frame'].pack(pady=0, padx=10, fill=ctk.BOTH, expand=True)
        self.widgets['text_frame'].grid_rowconfigure(0, weight=1)
        self.widgets['text_frame'].grid_columnconfigure(0, weight=1)
        self.widgets['text_frame'].grid_columnconfigure(1, weight=0)

        self.widgets['text_box'] = CustomText(self.widgets['text_frame'])
        self.widgets['text_box'].grid(row=0, column=0, sticky="nsew")

        self.widgets['scrollbar'] = ctk.CTkScrollbar(self.widgets['text_frame'])
        self.widgets['scrollbar'].grid(row=0, column=1, sticky="ns")
        self.widgets['text_box'].configure(yscrollcommand=self.widgets['scrollbar'].set)

    def initialize_entry_frame(self):
        self.widgets['entry_frame'] = ctk.CTkFrame(self.master)
        self.widgets['entry_frame'].pack(fill=ctk.X, padx=(5, 10), pady=10)

        self.widgets['entry'] = ctk.CTkTextbox(self.widgets['entry_frame'], height=100)
        self.widgets['entry'].pack(pady=0, padx=5, fill=ctk.X, expand=True)

        self.widgets['entry'].bind("<Return>", self.send_message_from_key)
        self.widgets['entry'].bind("<KeyRelease>", self.prevent_multiline)

    def prevent_multiline(self, event):
        if event.keysym == "Return":
            self.widgets['entry'].delete("insert", "end lineend")

    def send_message_from_key(self, event):
        if event.keysym == "Return":
            user_message = self.widgets['entry'].get("1.0", "end-1c").strip()
            if user_message:
                self.send_message(user_message)
                self.widgets['entry'].delete("1.0", "end")

    def send_message(self, user_message):
        self.stop_streaming = True
        if self.streaming_thread and self.streaming_thread.is_alive():
            self.streaming_thread.join()
        self.stop_streaming = False
        self.display_user_message(user_message)
        with self.response_queue.mutex:
            self.response_queue.queue.clear()
        self.streaming_thread = threading.Thread(target=stream_response, args=(self, user_message))
        self.streaming_thread.start()
        self.master.after(100, self.check_response_queue)

    def check_response_queue(self):
        try:
            response = self.response_queue.get(block=False)
            if response is None:
                self.display_response({"type": "text", "content": ""}, end_with_newline=True)
            else:
                if not self.stop_streaming:
                    self.display_response(response, end_with_newline=False)
            self.master.after(100, self.check_response_queue)
        except queue.Empty:
            self.master.after(100, self.check_response_queue)

    def display_user_message(self, user_message):
        with self.lock:
            self.widgets['text_box'].configure(state="normal")
            self.widgets['text_box'].insert('end', f"User: {user_message}\n", "user")
            self.widgets['text_box'].configure(state="disabled")
            self.widgets['text_box'].yview('end')

    def display_response(self, response, end_with_newline=False):
        with self.lock:
            self.widgets['text_box'].configure(state="normal")
            if response["type"] == "text":
                content = response['content'].strip()
                if content:  # Only display non-empty content
                    self.widgets['text_box'].insert('end', f"{content}\n", "assistant")
            elif response["type"] == "code":
                content = response['content'].strip()
                if content:  # Only display non-empty content
                    # Insert code with buttons
                    self.widgets['text_box'].insert_code(content, language=response['language'])
            elif response["type"] == "buttons":
                self.widgets['text_box'].insert('end', f"{response['content']}\n", "buttons")
            self.widgets['text_box'].configure(state="disabled")
            self.widgets['text_box'].yview('end')

    def display_key_points(self, key_points):
        """
        Display the key points from the scraped article in the GUI.
        """
        with self.lock:
            self.widgets['text_box'].configure(state="normal")
            self.widgets['text_box'].insert('end', "\n**Key Points from the Article:**\n", "assistant")
            for i, point in enumerate(key_points, 1):
                self.widgets['text_box'].insert('end', f"{i}. {point}\n", "assistant")
            self.widgets['text_box'].configure(state="disabled")
            self.widgets['text_box'].yview('end')

    def clear_chat(self, new_conversation=False):
        """
        Clear the chat history in the UI.
        """
        self.stop_streaming = True
        if self.streaming_thread and self.streaming_thread.is_alive():
            self.streaming_thread.join()
        with self.response_queue.mutex:
            self.response_queue.queue.clear()
        self.widgets['text_box'].configure(state="normal")
        self.widgets['text_box'].delete(1.0, "end")
        self.widgets['text_box'].configure(state="disabled")
        self.widgets['entry'].delete("1.0", "end")

    def stop_current_streaming(self):
        self.stop_streaming = True
        if self.streaming_thread and self.streaming_thread.is_alive():
            self.streaming_thread.join()