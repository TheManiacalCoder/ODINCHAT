from .message_parser import MessageParser
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def stream_response(chatbot_ui, user_message):
    """
    Stream the AI response to the GUI, ensuring code blocks are displayed simply with a black background.
    """
    try:
        # Process the user message and get the AI response
        if not hasattr(chatbot_ui.conversation_manager, 'process_query'):
            logging.error("ConversationManager does not have a 'process_query' method.")
            chatbot_ui.response_queue.put({"type": "text", "content": "Error: ConversationManager is not properly initialized."})
            chatbot_ui.response_queue.put(None)  # Signal end of response
            return

        response = chatbot_ui.conversation_manager.process_query(user_message)
        if not response:
            chatbot_ui.response_queue.put({"type": "text", "content": "Error: No response from the AI model."})
            chatbot_ui.response_queue.put(None)  # Signal end of response
            return

        # Parse the response into messages
        parser = MessageParser()
        parsed_messages = parser.parse_response(response)

        # Send parsed messages to the queue
        for message in parsed_messages:
            chatbot_ui.response_queue.put(message)

        chatbot_ui.response_queue.put(None)  # Signal end of response

    except Exception as e:
        logging.error(f"Error in stream_response: {str(e)}")
        chatbot_ui.response_queue.put({"type": "text", "content": f"Error: {str(e)}"})
        chatbot_ui.response_queue.put(None)  # Signal end of response