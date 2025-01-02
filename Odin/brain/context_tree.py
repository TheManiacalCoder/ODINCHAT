import os
import csv
from datetime import datetime

class ContextTreeNode:
    def __init__(self, context_id, context_name, parent=None):
        """
        Represents a node in the context tree.
        """
        self.context_id = context_id  # Unique ID for the context
        self.context_name = context_name  # Name of the context (e.g., "Topic A")
        self.parent = parent  # Parent node (None for the root)
        self.children = []  # List of child nodes
        self.messages = []  # Stores conversation history (user queries and AI responses)
        self.role = None  # Role for this context (e.g., "Assistant - Explain concepts")
        self.embeddings = []  # Word2Vec embeddings for messages in this context

    def add_child(self, child_node):
        """
        Add a child node to this node.
        """
        self.children.append(child_node)

    def add_message(self, message):
        """
        Add a message to this context.
        """
        self.messages.append(message)

    def get_full_context(self):
        """
        Retrieve the full conversation history for this context.
        """
        return self.messages

    def __repr__(self):
        """
        String representation of the node for debugging.
        """
        return f"ContextTreeNode(context_id={self.context_id}, context_name={self.context_name}, messages={len(self.messages)}, children={len(self.children)})"


class ContextTree:
    def __init__(self):
        """
        Represents the hierarchical context tree.
        """
        self.root = ContextTreeNode(context_id="root", context_name="Global Conversation")
        self.current_node = self.root  # Tracks the current context
        self.output_file = None  # File to which the context tree is appended

    def create_new_context(self, context_name, role=None):
        """
        Create a new context node and switch to it.
        """
        new_node = ContextTreeNode(
            context_id=f"context_{len(self.current_node.children)}",
            context_name=context_name,
            parent=self.current_node,
        )
        new_node.role = role  # Assign a role to the new context
        self.current_node.add_child(new_node)
        self.switch_context(new_node)
        return new_node

    def switch_context(self, context_node):
        """
        Switch the current context to the specified node.
        """
        self.current_node = context_node

    def get_current_context(self):
        """
        Retrieve the current context node.
        """
        return self.current_node

    def initialize_output_file(self, file_name):
        """
        Initialize the output CSV file in the same directory as engine.py.
        Creates a new file (or overwrites an existing one) every time the program starts.
        """
        # Get the directory of the engine.py file
        engine_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the full file path
        self.output_file = os.path.join(engine_dir, file_name)
        # Create or overwrite the file and write the header
        with open(self.output_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Timestamp",
                "Context ID",
                "Context Name",
                "Parent Context ID",
                "Role",
                "Message Role",
                "Message Content"
            ])
        print(f"Created new context tree CSV file at: {self.output_file}")

    def append_message_to_file(self):
        """
        Append the latest message in the current context to the CSV file.
        """
        if not self.output_file:
            raise ValueError("Output file not initialized. Call `initialize_output_file` first.")

        # Get the latest message in the current context
        if not self.current_node.messages:
            return  # No messages to append

        latest_message = self.current_node.messages[-1]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        context_id = self.current_node.context_id
        context_name = self.current_node.context_name
        parent_context_id = self.current_node.parent.context_id if self.current_node.parent else "None"
        role = self.current_node.role
        message_role = latest_message["role"]
        message_content = latest_message["content"]

        # Append the message to the CSV file
        with open(self.output_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                timestamp,
                context_id,
                context_name,
                parent_context_id,
                role,
                message_role,
                message_content
            ])
        print(f"Appended message to {self.output_file}: {message_role}: {message_content}")

    def __repr__(self):
        """
        String representation of the tree for debugging.
        """
        return self._print_tree(self.root, 0)

    def _print_tree(self, node, level):
        """
        Recursively print the tree structure.
        """
        result = "  " * level + repr(node) + "\n"
        for child in node.children:
            result += self._print_tree(child, level + 1)
        return result