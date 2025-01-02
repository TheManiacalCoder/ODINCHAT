import customtkinter as ctk
from .CustomText import CustomText

class ChatbotFields:
    def __init__(self, master, chatbot_ui):
        self.master = master
        self.chatbot_ui = chatbot_ui

        self.text_frame = ctk.CTkFrame(self.master)
        self.text_frame.pack(pady=10, padx=10, fill=ctk.BOTH, expand=True)

        self.text_box = CustomText(self.text_frame)
        self.text_box.pack(pady=10, padx=10, fill='both', expand=True)

        self.scrollbar = ctk.CTkScrollbar(self.text_frame, command=self.text_box.yview)
        self.scrollbar.pack(side='right', fill='y')

        self.text_box.config(yscrollcommand=self.scrollbar.set)

        self.entry_frame = ctk.CTkFrame(self.master)
        self.entry_frame.pack(fill=ctk.X, padx=10, pady=(0, 10))

        self.entry = ctk.CTkTextbox(self.entry_frame, height=100)
        self.entry.pack(pady=10, padx=10, fill=ctk.X, expand=True)
        self.entry.bind("<Return>", self.chatbot_ui.send_message_from_key)
        self.entry.bind("<KeyRelease>", self.chatbot_ui.prevent_multiline)