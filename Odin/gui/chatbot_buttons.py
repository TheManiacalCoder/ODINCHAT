# START OF FILE: C:\Users\Sean Craig\Desktop\AI Python Tools\Odin\gui\chatbot_buttons.py
import customtkinter as ctk
import os
import shutil

class ChatbotButtons:
    def __init__(self, master, chatbot_ui):
        self.master = master
        self.chatbot_ui = chatbot_ui

        # First Row: Update Model and Model Field
        self.first_row_frame = ctk.CTkFrame(self.master)
        self.first_row_frame.pack(expand=True, fill=ctk.X, padx=10, pady=(10, 0))

        # Update Model Button
        self.update_model_button = ctk.CTkButton(
            self.first_row_frame, 
            text="Update Model", 
            command=self.update_model, 
            width=180,
            height=50,
            fg_color="#000000",  
            corner_radius=0,     
            font=("Segoe UI", 15)  
        )
        self.update_model_button.pack(side=ctk.LEFT, padx=10, pady=10)

        # Text Field for Model Name
        self.model_name_entry = ctk.CTkEntry(
            self.first_row_frame, 
            font=("Segoe UI", 18),
            placeholder_text=self.chatbot_ui.conversation_manager.MODEL_NAME,
            height=50,  
            width=500   
        )
        self.model_name_entry.pack(side=ctk.LEFT, padx=(10, 10), pady=10, fill=ctk.X, expand=True)

        # Second Row: New Chat, Clear Chat, Update Memory, and Clear Memories
        self.second_row_frame = ctk.CTkFrame(self.master)
        self.second_row_frame.pack(expand=True, fill=ctk.X, padx=10, pady=(0, 10))

        # New Chat Button
        self.new_conversation_button = ctk.CTkButton(
            self.second_row_frame, 
            text="New Chat", 
            command=self.new_conversation, 
            width=180,
            height=50,
            fg_color="#000000",  
            corner_radius=0,     
            font=("Segoe UI", 15)  
        )
        self.new_conversation_button.pack(side=ctk.LEFT, padx=10, pady=10)

        # Clear Chat Button
        self.clear_chat_button = ctk.CTkButton(
            self.second_row_frame, 
            text="Clear Chat", 
            command=self.clear_chat, 
            width=180,
            height=50,
            fg_color="#000000",  
            corner_radius=0,     
            font=("Segoe UI", 15)  
        )
        self.clear_chat_button.pack(side=ctk.LEFT, padx=10, pady=10)

        # Update Memory Button
        self.file_picker_button = ctk.CTkButton(
            self.second_row_frame, 
            text="Update Memory", 
            command=self.pick_file, 
            width=180,
            height=50,
            fg_color="#000000",  
            corner_radius=0,     
            font=("Segoe UI", 15)  
        )
        self.file_picker_button.pack(side=ctk.LEFT, padx=10, pady=10)

        # Clear Memories Button
        self.clear_memories_button = ctk.CTkButton(
            self.second_row_frame, 
            text="Clear Memories", 
            command=self.clear_memories, 
            width=180,
            height=50,
            fg_color="#000000",  
            corner_radius=0,     
            font=("Segoe UI", 15)  
        )
        self.clear_memories_button.pack(side=ctk.LEFT, padx=10, pady=10)

    def new_conversation(self):
        self.chatbot_ui.conversation_manager.clear_conversation(new_conversation=True)
        self.chatbot_ui.clear_chat(new_conversation=True)

    def clear_chat(self):
        self.chatbot_ui.conversation_manager.clear_conversation(new_conversation=False)
        self.chatbot_ui.clear_chat(new_conversation=False)

    def clear_memories(self):
        memory_dir = self.chatbot_ui.conversation_manager.memory_dir
        if os.path.exists(memory_dir):
            for folder in os.listdir(memory_dir):
                folder_path = os.path.join(memory_dir, folder)
                if os.path.isdir(folder_path):
                    shutil.rmtree(folder_path)
            self.chatbot_ui.conversation_manager.init_conversation()
            self.chatbot_ui.clear_chat(new_conversation=True)
            self.chatbot_ui.widgets['text_box'].configure(state="normal")
            self.chatbot_ui.widgets['text_box'].insert('end', "All memories have been cleared.\n", "assistant")
            self.chatbot_ui.widgets['text_box'].configure(state="disabled")
            self.chatbot_ui.widgets['text_box'].yview('end')
        else:
            self.chatbot_ui.widgets['text_box'].configure(state="normal")
            self.chatbot_ui.widgets['text_box'].insert('end', "Memory directory not found.\n", "assistant")
            self.chatbot_ui.widgets['text_box'].configure(state="disabled")
            self.chatbot_ui.widgets['text_box'].yview('end')

    def update_model(self):
        new_model_name = self.model_name_entry.get().strip()
        if new_model_name:
            self.chatbot_ui.conversation_manager.set_model_name(new_model_name)
            self.chatbot_ui.clear_chat(new_conversation=False)
            self.chatbot_ui.widgets['text_box'].configure(state="normal")
            self.chatbot_ui.widgets['text_box'].insert('end', f"Model updated to {new_model_name}.\n", "assistant")
            self.chatbot_ui.widgets['text_box'].configure(state="disabled")
            self.chatbot_ui.widgets['text_box'].yview('end')

    def pick_file(self):
        self.chatbot_ui.conversation_manager.pick_file()
        self.chatbot_ui.clear_chat(new_conversation=False)
# END OF FILE: C:\Users\Sean Craig\Desktop\AI Python Tools\Odin\gui\chatbot_buttons.py