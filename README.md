# Chatbot for Web Traffic Logs

## Description

This project implements a chatbot that interacts with users based on web traffic log data. Using advanced language models and vector search technologies, the chatbot provides insights and answers questions derived from web traffic logs. The system is built with Python and integrates various tools to process and query log data effectively.

### Data Processing

1. **Data Loading**: The application loads web traffic log data from a CSV file using the `CSVLoader` class.
2. **Text Splitting**: The log data is split into chunks using the `RecursiveCharacterTextSplitter` to handle large amounts of text.
3. **Embedding**: The text chunks are embedded using `OpenAIEmbeddings` for vector representation.
4. **Vector Storage**: The embeddings are stored in a Pinecone vector store for efficient retrieval.

### Model Configuration

- **LLM**: The application uses OpenAI's `ChatOpenAI` model with `gpt-4` to generate responses.
- **Vector Store**: Pinecone is used as the vector store to retrieve relevant documents based on user queries.
- **Prompt Management**: The prompts are managed using `LangChain`'s prompt templates and memory management techniques.

### User Interface

The user interface is built using [Streamlit](https://streamlit.io), providing a simple and interactive platform for users to ask questions and receive responses. The interface includes:
- **Text Input Field**: For users to type their questions.
- **Submit Button**: To send the question to the chatbot.
- **Conversation History**: Displays the ongoing conversation between the user and the chatbot.

![image](https://github.com/user-attachments/assets/217a1725-4727-4e80-ae09-20617916e4f4)

## Installation

To get started with this project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/zehraozturkk/chatbot-web-traffic-logs.git
   cd chatbot-web-traffic-logs
2. **Create and Activate a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt

4. **Set Up Environment Variables**:
   ```bash
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key

5. **Run the Application**:
   ```bash
   streamlit run app.py

## Acknowledgments

- [LangChain](https://github.com/langchain/langchain) for providing the framework for language models and document retrieval.
- [Pinecone](https://www.pinecone.io) for vector search capabilities.
- [Streamlit](https://streamlit.io) for the interactive user interface.
- [OpenAI](https://www.openai.com) for providing the language model used in the application.

## Contact

For any questions or issues, please contact:

- **Zehra Öztürk** - [fzehraozturk34@gmail.com](mailto:fzehraozturk34@gmail.com)
