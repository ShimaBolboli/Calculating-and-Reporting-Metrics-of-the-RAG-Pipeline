from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from langchain import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_pinecone import PineconeVectorStore
from openai import OpenAI
from langchain_openai import ChatOpenAI
import streamlit as st
import openai

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
##openai_org_id = os.getenv("OPENAI_ORG_ID")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

openai.api_key = openai_api_key
##openai.organization = openai_org_id

# Load and split documents
loader = TextLoader('./book.txt')
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)  # Adjust chunk size as needed
docs = loader.load_and_split(text_splitter)
print ('$$$$$doc',docs)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Pinecone instance
##api_key = os.getenv('PINECONE_API_KEY')  # Ensure this is set in your .env file
pc = Pinecone(api_key=pinecone_api_key, environment='us-east-1')

index_name = "langchain-demo"

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
llm = ChatOpenAI(model_name=model_name)##organization="org-cO7NxVKMzgJ1h4pOwnv22ZIA")

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

st.title("📚 Book Recommendation Chatbot")

# Function for generating LLM response
def generate_response(query):
    result = rag_chain({"query": query})
    return result

# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm here to assist with your book recommendations. How can I help you today?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if user_input := st.chat_input("Ask your book-related question here"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Getting your answer..."):
                response = generate_response(user_input)
                st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
