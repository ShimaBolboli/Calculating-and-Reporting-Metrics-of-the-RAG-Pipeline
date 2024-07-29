# Calculating-and-Reporting-Metrics-of-the-RAG-Pipeline
# LangChain Chatbot Evaluation

This project demonstrates how to set up a Retrieval-Augmented Generation (RAG) pipeline using the LangChain framework, evaluate various metrics of a chatbot, and measure its performance. The evaluation includes context precision, recall, relevance, noise robustness, and more. 

## Features

- **Document Loading**: Loads and splits text documents.
- **Embeddings**: Uses the HuggingFace embeddings for text representation.
- **Vector Store**: Stores and retrieves document embeddings using Pinecone.
- **Chatbot**: Uses OpenAI's GPT model for generating responses.
- **Evaluation Metrics**: Measures context precision, recall, relevance, and various other performance metrics.

## Setup

### Prerequisites

- Python 3.7+
- Required Python packages: `langchain_community`, `langchain_huggingface`, `langchain_core`, `langchain_pinecone`, `langchain_openai`, `openai`, `pinecone`, `scikit-learn`, `numpy`, `python-dotenv`

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/your-repo.git
    cd your-repo
    ```

2. **Install the required packages:**

    ```bash
    pip install langchain_community langchain_huggingface langchain_core langchain_pinecone langchain_openai openai pinecone-client scikit-learn numpy python-dotenv
    ```

3. **Create a `.env` file in the root directory with the following content:**

    ```env
    OPENAI_API_KEY=your_openai_api_key
    PINECONE_API_KEY=your_pinecone_api_key
    ```

## Usage

1. **Prepare your documents**: Place the text document you want to use in the root directory and name it `book.txt`.

2. **Run the script**:

    ```bash
    python your_script.py
    ```

3. **Metrics Output**: The script will output the evaluation metrics for the chatbot. 

### Explanation

- **Loading and Splitting Documents**: Uses `TextLoader` and `CharacterTextSplitter` to load and split text documents.
- **Initializing Embeddings**: Utilizes `HuggingFaceEmbeddings` to convert text into vector representations.
- **Pinecone Setup**: Initializes Pinecone for storing and retrieving document embeddings.
- **ChatOpenAI Model**: Uses OpenAI's GPT model to generate responses based on user queries.
- **Evaluation Functions**: Calculates various metrics like context precision, recall, relevance, noise robustness, and more.

### Metrics Functions

- `calculate_context_precision`: Measures how accurately predicted labels match the true labels in terms of context similarity.
- `calculate_context_recall`: Measures how well the true labels are covered by the predicted labels.
- `calculate_context_relevance`: Evaluates the relevance of the generated answers to the true labels.
- `calculate_context_entity_recall`: Assesses the recall of relevant entities in the context.
- `evaluate_noise_robustness`: Measures the chatbot’s ability to handle noisy or irrelevant queries.
- `calculate_faithfulness`: Evaluates the similarity between generated answers and true answers.
- `calculate_answer_relevance`: Measures the relevance of the generated answers to the true labels.
- `evaluate_information_integration`: Assesses how well the generated answers integrate information from the true contexts.
- `evaluate_counterfactual_robustness`: Tests the chatbot’s robustness against counterfactual or contradictory queries.
- `evaluate_negative_rejection`: Measures the system's ability to handle and reject negative or inappropriate queries.
- `calculate_latency`: Measures the average response time of the chatbot.

## Notes

- Ensure you have valid API keys for OpenAI and Pinecone in the `.env` file.
- Customize the example queries, true labels, and other parameters as needed for your specific use case.





### YouTube Link
------------------------------------------------------------
https://youtu.be/tyqQT_Nj0Ak
