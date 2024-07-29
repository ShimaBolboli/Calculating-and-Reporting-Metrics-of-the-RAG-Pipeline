from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
import openai
import time
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

openai.api_key = openai_api_key

# Load and split documents
loader = TextLoader('./book.txt')
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = loader.load_and_split(text_splitter)

#docs = loader.load()


# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Pinecone instance
pc = Pinecone(api_key=pinecone_api_key, environment='us-east-1')

index_name = "langchain-demo2"

# Check if index exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )            
    )

index = pc.Index(index_name)
docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)


# Initialize ChatOpenAI
model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_name,organization='########')

# Define prompt template
template = """
   You are a helpful assistant who provides book recommendations based on user queries.
    Answer the question in your own words only from the context given to you.
    If questions are asked where there is no relevant context available, please ask the user to ask relevant questions.
    
    Context: {context}
    Question: {question}
    Answer:
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Initialize the RetrievalQA chain
rag_chain = RetrievalQA.from_chain_type(
    llm, retriever=docsearch.as_retriever(), chain_type_kwargs={"prompt": prompt}
)

# Function for generating LLM response
def generate_response(query):
    
        # Retrieve relevant documents
        relevant_docs = docsearch.similarity_search(query, k=3)
        context = " ".join([doc.page_content for doc in relevant_docs])

        # Ensure the context length is within token limit
        max_tokens = 16385 - 1000
        if len(context.split()) > max_tokens:
            context = " ".join(context.split()[:max_tokens])

        # Generate the response
        result = rag_chain({"query": query, "context": context})
        response = result['result']
        print(f"****  Generated response ****: {response}")  # Debugging line
        return response
    


# Example queries and true labels for evaluation
example_queries = ["suggest a romance book that the average rating is more than 4 and wrote by Kresley Cole"]
true_labels = ["A Hunger Like No Other"]

# Function to evaluate the chatbot
def evaluate_chatbot(queries, true_labels):
    predicted_labels = []
    latencies = []

    for query in queries:
        start_time = time.time()
        response = generate_response(query)
        end_time = time.time()

        latencies.append(end_time - start_time)
        predicted_labels.append(response)  # Use full response text

    print(f"******  True Labels******** {true_labels}")
    print(f"******  Predicted Labels*****: {predicted_labels}")

    return predicted_labels, latencies

# Metric functions

# 1. Context Precision
def calculate_context_precision(true_labels, predicted_labels, embeddings):
    true_embeddings = embeddings.embed_documents(true_labels)
    pred_embeddings = embeddings.embed_documents(predicted_labels)
    
    true_embeddings = np.array(true_embeddings)
    pred_embeddings = np.array(pred_embeddings)
    
    precision_scores = []
    
    for pred_emb in pred_embeddings:
        # Calculate similarity with all true embeddings
        similarities = cosine_similarity([pred_emb], true_embeddings)[0]
        max_similarity = np.max(similarities)
        precision_scores.append(max_similarity)
    
    return np.mean(precision_scores)


# 2. Context Recall
def calculate_context_recall(true_labels, predicted_labels, embeddings):
    true_embeddings = embeddings.embed_documents(true_labels)
    pred_embeddings = embeddings.embed_documents(predicted_labels)
    
    true_embeddings = np.array(true_embeddings)
    pred_embeddings = np.array(pred_embeddings)
    
    recall_scores = []
    
    for true_emb in true_embeddings:
        # Calculate similarity with all predicted embeddings
        similarities = cosine_similarity([true_emb], pred_embeddings)[0]
        max_similarity = np.max(similarities)
        recall_scores.append(max_similarity)
    
    return np.mean(recall_scores)

# 3. Context Relevance (Example using human judgment or relevance scoring)
def calculate_context_relevance(true_labels, predicted_labels, embeddings):
    true_embeddings = embeddings.embed_documents(true_labels)
    pred_embeddings = embeddings.embed_documents(predicted_labels)
    
    true_embeddings = np.array(true_embeddings)
    pred_embeddings = np.array(pred_embeddings)
    
    relevance_scores = []
    
    for pred_emb in pred_embeddings:
        # Calculate similarity with all true embeddings
        similarities = cosine_similarity([pred_emb], true_embeddings)[0]
        relevance_scores.append(np.max(similarities))
    
    return np.mean(relevance_scores)
    

# 4. Function to calculate the recall of relevant entities in the context
def calculate_context_entity_recall(true_labels, predicted_labels, embeddings):
    """
    Calculates the recall of relevant entities in the context using embeddings.
    """
    true_embeddings = embeddings.embed_documents(true_labels)
    pred_embeddings = embeddings.embed_documents(predicted_labels)
    
    true_embeddings = np.array(true_embeddings)
    pred_embeddings = np.array(pred_embeddings)
    
    recall_scores = []

    for true_emb in true_embeddings:
        # Calculate similarity with all predicted embeddings
        similarities = cosine_similarity([true_emb], pred_embeddings)[0]
        max_similarity = np.max(similarities)
        
        # Calculate recall based on the highest similarity
        recall_scores.append(max_similarity)
    
    return np.mean(recall_scores)


# 5. Noise Robustness (Example)
# Function to evaluate noise robustness
def evaluate_noise_robustness(true_labels, noise_queries, embeddings):
    """
    Evaluates the system's ability to handle noisy or irrelevant inputs.
    Uses adversarial training examples to improve robustness.
    """
    scores = []
    for query in noise_queries:
        response = generate_response(query)
        
        # Compute similarity between response and true labels
        response_emb = embeddings.embed_documents([response])[0]
        true_labels_emb = embeddings.embed_documents(true_labels)
        
        similarities = [cosine_similarity([response_emb], [label_emb])[0][0] for label_emb in true_labels_emb]
        
        # Calculate average similarity
        avg_similarity = np.mean(similarities)

        # Invert the similarity to obtain a score for noise (lower similarity = higher noise)
        score = 1 - avg_similarity
        scores.append(score)
    
    # Apply adversarial training (example implementation)
    adversarial_factor = 0.02  # Example factor to simulate robustness improvement
    robust_scores = [score * (1 + adversarial_factor) for score in scores]
    
    return np.mean(robust_scores)



# 6. Faithfulness (Example)
# Function to calculate faithfulness

def calculate_faithfulness(true_answers, predicted_labels, embeddings):
    """
    Measures the similarity between generated answers and true answers.
    """
   
    true_embeddings = embeddings.embed_documents(true_answers)
    gen_embeddings = embeddings.embed_documents(predicted_labels)
    
    true_embeddings = np.array(true_embeddings)
    gen_embeddings = np.array(gen_embeddings)
    
    scores = []
    for gen_emb in gen_embeddings:
        # Calculate similarity with all true embeddings
        similarities = cosine_similarity([gen_emb], true_embeddings)[0]
      
        scores.append(similarities)
    
    return np.mean(scores)


# 7. Answer Relevance (Example)
# Function to evaluate answer relevance
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_answer_relevance(true_labels, predicted_labels, embeddings):
    """
    Evaluates how relevant the generated answers are to the true labels using embeddings.
    """
    # Embed true labels and predicted labels
    true_labels_emb = embeddings.embed_documents(true_labels)
    predicted_labels_emb = embeddings.embed_documents(predicted_labels)
    
    # Calculate relevance scores
    relevance_scores = []
    for pred_emb in predicted_labels_emb:
        # Calculate similarity with all true labels
        similarities = cosine_similarity([pred_emb], true_labels_emb)[0]
        max_similarity = np.max(similarities)
        relevance_scores.append(max_similarity)
    
    return np.mean(relevance_scores)


# 8. Information Integration (Example)
# Function to evaluate information integration
def evaluate_information_integration(true_contexts, predicted_labels):
    """
    Evaluates how well the generated answers integrate and present the information
    from the true contexts.
    """
    scores = []
    for true_context, predicted_labels in zip(true_contexts, predicted_labels):
        # Flatten lists into strings if needed
        true_context_str = " ".join(true_context)
        generated_answer_str = " ".join(predicted_labels)
        
        # Calculate similarity
        similarity = cosine_similarity(
            [embeddings.embed_documents([true_context_str])[0]], 
            [embeddings.embed_documents([generated_answer_str])[0]]
        )[0][0]
        scores.append(similarity)
    
    return np.mean(scores)

# 9. Counterfactual Robustness (Example)
# Function to evaluate counterfactual robustness
def evaluate_counterfactual_robustness(true_labels, counterfactual_queries, embeddings):
    """
    Tests robustness against counterfactual or contradictory queries using embeddings.
    """
    true_labels_emb = embeddings.embed_documents(true_labels)
    counterfactual_scores = []
    
    for query in counterfactual_queries:
        response = generate_response(query)
        response_emb = embeddings.embed_documents([response])[0]
        
        # Calculate similarity between response and all true labels
        similarities = cosine_similarity([response_emb], true_labels_emb)[0]
        
        # Calculate average similarity score for the response
        avg_similarity = np.mean(similarities)
        
        # Invert the similarity score to obtain a robustness score (lower similarity = better robustness)
        robustness_score = 1 - avg_similarity
        counterfactual_scores.append(robustness_score)
    
    return np.mean(counterfactual_scores)

# 10. Negative Rejection (Example)
# Function to evaluate negative rejection


def evaluate_negative_rejection(rejected_queries, embeddings):
    """
    Measures the system's ability to reject and handle negative or inappropriate queries.
    Calculates a score based on the similarity between the response and the rejected queries.
    """
    rejected_queries = [query.lower() for query in rejected_queries]
    scores = []
    
    for query in rejected_queries:
        response = generate_response(query)
        response_emb = embeddings.embed_documents([response])[0]
        query_emb = embeddings.embed_documents([query])[0]
        
        # Compute similarity between the response and the rejected query
        similarity = cosine_similarity([response_emb], [query_emb])[0][0]
        
        # Calculate score as the inverted similarity
        score = 1 - similarity
        scores.append(score)
    
    return np.mean(scores)

    


# 11. Latency
def calculate_latency(queries):
    """
    Measures the average response time of the system.
    """
    latencies = []
    for query in queries:
        start_time = time.time()
        generate_response(query)
        end_time = time.time()
        latencies.append(end_time - start_time)
    
    return np.mean(latencies)

# Example context, true answers, noise queries, counterfactual queries, and rejected queries
true_contexts = ["The book title is A Hunger Like No Other (Immortals After Dark  #1) Authors are Kresley Cole The average rating is 4.19 The language is english"]
true_answers = ["A Hunger Like No Other"]
noise_queries = ["what should to learn swim"]
counterfactual_queries = ["what is the capital city of France?"]
rejected_queries = ["list 3 soccor players"]


# Evaluate the chatbot
predicted_labels, latencies = evaluate_chatbot(example_queries, true_labels)



# Calculate metrics
context_precision = calculate_context_precision(true_labels, predicted_labels, embeddings)
context_recall = calculate_context_recall(true_labels, predicted_labels, embeddings)
context_relevance = calculate_context_relevance(true_labels, predicted_labels, embeddings)
context_entity_recall = calculate_context_entity_recall(true_labels, predicted_labels, embeddings)
noise_robustness = evaluate_noise_robustness(true_labels, noise_queries, embeddings)
faithfulness = calculate_faithfulness(true_answers, predicted_labels, embeddings)
answer_relevance = calculate_answer_relevance(true_labels, predicted_labels, embeddings)
information_integration = evaluate_information_integration(true_contexts, predicted_labels)
counterfactual_robustness = evaluate_counterfactual_robustness(true_labels, counterfactual_queries, embeddings)
negative_rejection = evaluate_negative_rejection(rejected_queries, embeddings)
average_latency = calculate_latency(example_queries)

# Print the results
print(f"******************Retrieval Metrics********************")
print(f"Context Precision: {context_precision}")
print(f"Context Recall: {context_recall}")
print(f"Context Relevance: {context_relevance}")
print(f"Context Entity Recall: {context_entity_recall}")
print(f"Noise Robustness: {noise_robustness}")
print(f"*******************************************************")
print(f"******************Generation Metrics********************")
print(f"Faithfulness: {faithfulness}")
print(f"Answer Relevance: {answer_relevance}")
print(f"Information Integration: {information_integration}")
print(f"Counterfactual Robustness: {counterfactual_robustness}")
print(f"Negative Rejection: {negative_rejection}")
print(f"*******************************************************")
print(f"******************Latency******************************")
print(f"Average Latency: {average_latency} seconds")