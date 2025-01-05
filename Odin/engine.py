import customtkinter as ctk
from gui.app import run_gui_wrapper
from brain.ai_memory import conversation_manager
from brain.agenticreason import AgenticReasoner
from config import OPEN_ROUTER_API_KEY, MODEL_NAME

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Set the OpenRouter API key and model name
conversation_manager.set_openrouter_api_key(OPEN_ROUTER_API_KEY)
conversation_manager.set_model_name(MODEL_NAME)

agentic_reasoner = AgenticReasoner(api_key=OPEN_ROUTER_API_KEY, model_name=MODEL_NAME)

def run_engine():
    root = ctk.CTk()
    root.title("Odin")

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    window_width = int(screen_width * 0.25)
    window_height = int(screen_height * 0.85)

    window_x = (screen_width - window_width) // 2
    window_y = (screen_height - window_height) // 2

    root.geometry(f"{window_width}x{window_height}+{window_x}+{window_y}")
    root.configure(fg_color="#000000")

    container = ctk.CTkFrame(root, fg_color="#000000", border_width=0)
    container.pack(padx=20, pady=20, fill=ctk.BOTH, expand=True)

    # Pass the ChatbotUI instance to the ConversationManager
    chatbot_ui = run_gui_wrapper(container)
    conversation_manager.chatbot_ui = chatbot_ui  # Ensure this line is present

    root.mainloop()

if __name__ == "__main__":
    run_engine()