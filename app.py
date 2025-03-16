import os
from dotenv import load_dotenv
from chatbot import DataChatbot

# Load environment variables from .env file
load_dotenv()

def main():
    """
    Main function to run the data chatbot CLI
    """
    print("=== Data Analysis Chatbot ===")
    print("Type 'exit' to quit, 'load <file_path>' to load a dataset")

    # Initialize the chatbot
    try:
        chatbot = DataChatbot()
        print("Chatbot initialized successfully!")
    except ValueError as e:
        print(f"Error: {e}")
        print("Make sure you have set your OPENAI_API_KEY in the .env file")
        return

    # Main interaction loop
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()

        # Check for exit command
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        # Check for load command
        if user_input.lower().startswith('load '):
            file_path = user_input[5:].strip()
            response = chatbot.load_dataset(file_path)
            print(f"Chatbot: {response}")
            continue

        # Regular question - ask the chatbot
        if user_input:
            response = chatbot.ask(user_input)
            print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()