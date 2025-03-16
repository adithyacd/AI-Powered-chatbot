import os
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from data_adapter import DataAdapter

class DataChatbot:
    """
    A chatbot that can answer questions about loaded data
    and maintain conversation context.
    """
    def __init__(self, api_key=None):
        """
        Initialize the chatbot.

        Args:
            api_key (str, optional): OpenAI API key. If not provided,
                                    will look for OPENAI_API_KEY in environment.
        """
        # Use provided API key or get from environment
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        # Initialize the language model
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",  # You can use gpt-4 for better results
            temperature=0.2,  # Lower temperature for more factual responses
            api_key=self.api_key
        )

        # Create conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Create the conversation chain
        self.conversation = self._create_conversation_chain()

        # Data related attributes
        self.data_adapter = DataAdapter()
        self.current_data = None
        self.data_description = None

    def _create_conversation_chain(self):
        """
        Create the conversation chain with the appropriate prompt template.

        Returns:
            ConversationChain: The configured conversation chain.
        """
        # Create a prompt template that includes data context and chat history
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
             You are a helpful data analysis assistant. You help users explore and understand
             their data through natural conversation. When responding to questions about data,
             use the data context provided to give accurate, clear answers.

             If you're asked about data that hasn't been loaded yet, politely ask the user to
             load a dataset first.

             Current data context: {data_context}
             """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # Create and return the conversation chain
        return ConversationChain(
            llm=self.llm,
            prompt=prompt,
            memory=self.memory,
            verbose=False
        )

    def load_dataset(self, file_path):
        """
        Load a dataset from the specified file path.

        Args:
            file_path (str): Path to the data file.

        Returns:
            str: Success message
        """
        try:
            # Load the data
            self.current_data = self.data_adapter.load_data(file_path)

            # Create a data description for context
            if isinstance(self.current_data, pd.DataFrame):
                self.data_description = f"""
                Dataset loaded from: {file_path}
                Shape: {self.current_data.shape[0]} rows, {self.current_data.shape[1]} columns
                Columns: {', '.join(self.current_data.columns)}
                """
            else:
                self.data_description = f"Text data loaded from: {file_path}"

            return f"Dataset loaded successfully: {self.data_description}"

        except Exception as e:
            return f"Error loading dataset: {str(e)}"

    def ask(self, question):
        """
        Ask a question about the loaded data.

        Args:
            question (str): The user's question.

        Returns:
            str: The chatbot's response
        """
        # Prepare the data context
        data_context = "No data loaded yet." if self.current_data is None else self.data_description

        # Get the response from the conversation chain
        response = self.conversation.predict(
            input=question,
            data_context=data_context
        )

        return response