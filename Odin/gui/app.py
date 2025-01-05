from .chatbot_ui import ChatbotUI

def run_gui_wrapper(container):
    chatbot_ui = ChatbotUI(container)
    return chatbot_ui  # Ensure this line is present