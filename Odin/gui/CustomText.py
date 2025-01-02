import tkinter as tk  # Add this import statement
import tkinter.filedialog as filedialog
import pyperclip
import sys

class CustomText(tk.Text):
    def __init__(self, master=None, **kw):
        super().__init__(master, **kw)
        self.configure(
            bg="#282828",
            fg="#FFFFFF",
            insertbackground="#FFFFFF",
            padx=0,
            pady=0,
            highlightthickness=0,
            relief='flat',
            wrap='word'
        )

        self.tag_configure(
            "user",
            background="#343434",
            foreground="#FFFFFF",
            font=("Segoe UI", 12),
            lmargin1=50,
            lmargin2=50,
            rmargin=100,
            wrap='word',
            spacing1=10,
            spacing2=0,
            spacing3=10
        )

        self.tag_configure(
            "assistant",
            background="#1E1E1E",
            foreground="#FFFFFF",
            font=("Segoe UI", 12),
            lmargin1=50,
            lmargin2=50,
            rmargin=100,
            wrap='word',
            spacing1=10,
            spacing2=0,
            spacing3=10
        )

        self.tag_configure(
            "code",
            background="#000000",
            foreground="#FFFFFF",
            font=("Courier New", 12),
            lmargin1=50,
            lmargin2=50,
            rmargin=100,
            wrap='word',
            spacing1=10,
            spacing2=0,
            spacing3=10
        )

        self.tag_configure(
            "buttons",
            background="#000000",
            foreground="#0078D7",
            font=("Courier New", 12, "underline"),
            lmargin1=50,
            lmargin2=50,
            rmargin=100,
            wrap='none',
            spacing1=10,
            spacing2=0,
            spacing3=10
        )

        self.tag_bind("buttons", "<Button-1>", self.handle_button_click)
        self.tag_bind("buttons", "<Enter>", self.on_button_enter)
        self.tag_bind("buttons", "<Leave>", self.on_button_leave)

    def handle_button_click(self, event):
        index = self.index("@%s,%s" % (event.x, event.y))
        print(f"Clicked at index: {index}")

        # Find the range of the "buttons" tag at the clicked position
        button_start = self.index(f"{index} linestart")
        button_end = self.index(f"{index} lineend")
        
        # Get the full text within the "buttons" tag
        button_text = self.get(button_start, button_end).strip()
        print(f"Button text: {button_text}")

        # Find the adjacent code block
        code_start, code_end = self.tag_prevrange("code", index)
        if code_start and code_end:
            code_content = self.get(code_start, code_end).strip()
            print(f"Code content: {code_content}")

            # Extract the language from the code block (if specified)
            language = self.get_language_from_code_block(code_start)
            print(f"Detected language: {language}")

            # Remove the backtick lines from the code content
            cleaned_code_content = self.remove_backtick_lines(code_content)
            print(f"Cleaned code content: {cleaned_code_content}")

            # Check which button was clicked
            if "Copy" in button_text and self.is_click_on_word(event, "Copy"):
                self.copy_code_to_clipboard(cleaned_code_content)
            elif "Save" in button_text and self.is_click_on_word(event, "Save"):
                self.save_code_to_file(cleaned_code_content, language)
        else:
            print("Error: Could not find the adjacent code block.")

    def is_click_on_word(self, event, word):
        """
        Check if the click occurred on a specific word.
        """
        index = self.index("@%s,%s" % (event.x, event.y))
        clicked_word = self.get(index + " wordstart", index + " wordend").strip()
        return clicked_word == word

    def get_language_from_code_block(self, code_start):
        """
        Extract the language from the code block (if specified).
        """
        # Get the first line of the code block
        first_line = self.get(code_start, f"{code_start} lineend").strip()
        if first_line.startswith("```"):
            # Extract the language (e.g., ```python -> "python")
            language = first_line[3:].strip()
            return language if language else None
        return None

    def remove_backtick_lines(self, code_content):
        """
        Remove the backtick lines from the code content.
        """
        lines = code_content.splitlines()
        cleaned_lines = [line for line in lines if not line.strip().startswith("```")]
        return "\n".join(cleaned_lines)

    def copy_code_to_clipboard(self, code_content):
        pyperclip.copy(code_content)
        print("Codeblock Copied!")

    def save_code_to_file(self, code_content, language=None):
        """
        Save the code block to a file with the appropriate file extension based on the language.
        """
        # Map languages to file extensions
        language_to_extension = {
            "python": ".py",
            "javascript": ".js",
            "java": ".java",
            "html": ".html",
            "css": ".css",
            "c": ".c",
            "cpp": ".cpp",
            "bash": ".sh",
            "sql": ".sql",
            # Add more mappings as needed
        }

        # Default to .txt if no language is specified
        file_extension = language_to_extension.get(language, ".txt")

        # Open a file save dialog with the appropriate file extension
        file_path = filedialog.asksaveasfilename(
            defaultextension=file_extension,
            filetypes=[(f"{language or 'Text'} Files", f"*{file_extension}"), ("All Files", "*.*")]
        )

        if file_path:
            with open(file_path, "w") as file:
                file.write(code_content)
            print(f"Codeblock saved to {file_path}")

    def insert_code(self, code, language=None):
        """
        Insert a code block with "Copy" and "Save" buttons, excluding the backtick lines.
        """
        # Insert the code content without the backtick lines
        self.insert('end', f"{code}\n", "code")
        # Insert the "Copy" and "Save" buttons
        self.insert('end', "[Copy] [Save]\n", ("buttons",))
        self.mark_set("insert", "end-2c")

    def on_button_enter(self, event):
        self.config(cursor="hand2")

    def on_button_leave(self, event):
        self.config(cursor="")