import os
import tkinter as tk
from tkinter import filedialog
import logging
import chardet
import PyPDF2
import docx
import csv
import threading
import customtkinter as ctk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FilePicker:
    def __init__(self, conversation_manager, file_picker_button):
        """
        Initialize the FilePicker with a reference to the ConversationManager and the file picker button.
        """
        self.conversation_manager = conversation_manager
        self.file_picker_button = file_picker_button  # Pass the button from ChatbotButtons
        self.progress_bar = None

    def pick_file(self):
        """
        Open a file dialog to select a file, read its content, and process it as a message.
        """
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title="Select a file to upload")
        if file_path:
            try:
                # Convert the button to a progress bar
                self.file_picker_button.configure(text="Processing...", fg_color="#000000", hover=False)
                self.file_picker_button.update()

                # Process the file in a separate thread
                threading.Thread(target=self.process_file, args=(file_path,)).start()
            except Exception as e:
                logging.error(f"Error processing file: {str(e)}")
                # Display the error message in the chatbot UI
                self.conversation_manager.chatbot_ui.widgets['text_box'].configure(state="normal")
                self.conversation_manager.chatbot_ui.widgets['text_box'].insert('end', f"Error processing file: {str(e)}\n", "assistant")
                self.conversation_manager.chatbot_ui.widgets['text_box'].configure(state="disabled")
                self.conversation_manager.chatbot_ui.widgets['text_box'].yview('end')
                self.restore_button()

    def process_file(self, file_path):
        """
        Process the file into separate chunks, generate embeddings, and save them to the database.
        """
        try:
            # Read the file content
            content = self.read_file_content(file_path)
            file_name = os.path.basename(file_path)

            # Split the content into chunks
            total_words = len(content.split())
            chunk_size = max(1, len(content.split()) // 10)  # Split into 10 chunks
            chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]

            # Add file-level identifiers to the first and last chunks
            if chunks:
                # Add start identifier to the first chunk
                chunks[0] = f"--- Start of File: {file_name} ---\n{chunks[0]}"
                # Add end identifier to the last chunk
                chunks[-1] = f"{chunks[-1]}\n--- End of File: {file_name} ---"

            # Update progress bar for gathering chunks
            for i, _ in enumerate(chunks):
                progress = (i + 1) / len(chunks) * 0.5  # First 50% for gathering
                self.file_picker_button.configure(text=f"Processing... {int(progress * 100)}%")
                self.file_picker_button.update()

            # Stage 2: Generate embeddings and save chunks to the database
            self.conversation_manager.save_chunks_to_db(chunks, file_name)
            self.file_picker_button.configure(text="Processing... 100%")
            self.file_picker_button.update()
        finally:
            # Restore the button after processing is complete
            self.restore_button()

    def restore_button(self):
        """
        Restore the file picker button after processing is complete.
        """
        self.file_picker_button.configure(text="Update Memory", fg_color="#000000", hover=True)
        self.file_picker_button.update()

    def detect_encoding(self, file_path):
        """
        Detect the encoding of a file.
        """
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            return result['encoding']

    def read_file_content(self, file_path):
        """
        Read the content of a file based on its extension.
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.pdf':
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                content = ""
                for page in reader.pages:
                    content += page.extract_text()
            return content
        elif file_extension == '.docx':
            doc = docx.Document(file_path)
            content = "\n".join([para.text for para in doc.paragraphs])
            return content
        elif file_extension == '.txt':
            encoding = self.detect_encoding(file_path)
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
            return content
        elif file_extension == '.csv':
            encoding = self.detect_encoding(file_path)
            with open(file_path, 'r', encoding=encoding) as file:
                reader = csv.reader(file)
                content = "\n".join([",".join(row) for row in reader])
            return content
        elif file_extension in ['.py', '.js', '.java', '.html', '.css', '.cpp', '.c', '.sh', '.sql']:
            encoding = self.detect_encoding(file_path)
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
            return content
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")