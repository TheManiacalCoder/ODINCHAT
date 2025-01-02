# START OF FILE: C:\Users\Sean Craig\Desktop\AI Python Tools\Odin 2\gui\message_parser.py

# ==================================================
# CONFIGURATION VARIABLES (Edit these as needed)
# ==================================================
INCLUDE_BACKTICKS = False  # True/False: Include the backtick lines (```) in the code block content
INCLUDE_LANGUAGE = True    # True/False: Include the language specification (e.g., "python") in the code block
INCLUDE_CODE_CONTENT = True  # True/False: Include the actual code content in the code block

# ==================================================
# MESSAGE PARSER CLASS
# ==================================================

class MessageParser:
    def __init__(self):
        """
        Initialize the MessageParser with the global configuration variables.
        """
        self.buffer = ""
        self.in_code_block = False
        self.code_language = None
        self.code_block = []

    def parse_response(self, response):
        parsed_messages = []
        # Split the response into lines
        lines = response.split('\n')
        for line in lines:
            if line.strip().startswith('```'):
                # Toggle code block state
                if self.in_code_block:
                    # End of code block
                    code_content = self._build_code_block_content()
                    if code_content:
                        parsed_messages.append({
                            "type": "code",
                            "content": code_content,
                            "language": self.code_language
                        })
                    self.in_code_block = False
                    self.code_language = None
                    self.code_block = []
                else:
                    # Start of code block
                    self.in_code_block = True
                    # Extract language if present
                    lang_part = line.strip().lstrip('```')
                    self.code_language = lang_part if lang_part else None
            elif self.in_code_block:
                # Add line to the current code block
                self.code_block.append(line)
            else:
                # Add text line
                if line.strip():
                    parsed_messages.append({
                        "type": "text",
                        "content": line.strip()
                    })
        return parsed_messages

    def _build_code_block_content(self):
        """
        Build the code block content based on the selected options.

        Returns:
            str: The formatted code block content.
        """
        content = []
        if INCLUDE_CODE_CONTENT and self.code_block:
            if INCLUDE_BACKTICKS:
                # Add backticks if required
                content.append('```' + (self.code_language or ''))
                content.extend(self.code_block)
                content.append('```')
            else:
                content.extend(self.code_block)
        return "\n".join(content) if content else None

# END OF FILE: C:\Users\Sean Craig\Desktop\AI Python Tools\Odin 2\gui\message_parser.py