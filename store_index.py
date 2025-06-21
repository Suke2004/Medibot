from src.helper import load_pdf_file, text_split, download_huggingface_model
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

exracted_data = load_pdf_file("Data/")
text_chunks = text_split(exracted_data)
embeddings = download_huggingface_model()

pc = Pinecone(api_keys = PINECONE_API_KEY)

index_name = "medibot" #Use a unique name for your index

# Use this if you want to create a new index
pc.create_index(
        name=index_name,
        dimension = 384,
        metric="cosine",
        spec = ServerlessSpec(
            cloud = "aws",
            region = "us-east-1"
        )
)

print("Index", index_name, "created successfully.")
# If you want to connect to an existing index, comment the above code and uncomment the below line
# pc.connect(index_name=index_name)
# Use this if you want to connect to an existing index
docsearch = PineconeVectorStore.from_documents(
    documents = text_chunks,
    index_name = index_name,
    embedding = embeddings,
)