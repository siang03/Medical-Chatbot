from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os


# Initialize flask
app = Flask(__name__)

load_dotenv()

# Load API KEYS
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

# Load existing embeddings
embeddings  = download_hugging_face_embeddings()

index_name = "medical-chatbot"
# Embed each chunk and upsert the embeddings into your Pinecone Index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Initialize the chain
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# Initialize the Gemini Model
chatModel = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=GEMINI_API_KEY,
    generation_config={
        "temperature": 0.3,         # factual answers
        "max_output_tokens": 150     # ~3 sentences
    }
)

prompt = ChatPromptTemplate.from_messages(
    {
        ("system", system_prompt),
        ("human", "{input}"),
    }
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Configure Routing

# Home
@app.route("/")
def index():
    return render_template('chat.html')

# Retrieve User Inputs and Provide Response Based On The Input
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print("Input : ", input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port = 8080, debug = True)