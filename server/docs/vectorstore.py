import os
import time
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone  # for later querying if needed

# Load environment variables from .env
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")  # e.g. chatbot

# Directory for uploaded files
UPLOADS_DIR = "./uploaded_docs"
os.makedirs(UPLOADS_DIR, exist_ok=True)

# ----------------- PINECONE SETUP -----------------

# Your existing Pinecone index "chatbot" is 1024-dim (llama-text-embed-v2).
# We'll use a FREE 1024-dim HuggingFace model: BAAI/bge-large-en
EMBED_DIM = 1024

# Init Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region="us-east-1")


# List existing indexes
existing_indexes = pc.list_indexes()
print("Existing indexes:", existing_indexes)

# Extract names
index_names = [idx["name"] for idx in existing_indexes]

# Create index if it does not exist (with 1024-dim to match the embedding model)
if PINECONE_INDEX_NAME not in index_names:
    print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}' with dim={EMBED_DIM}")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=spec,
    )
    # wait until index is ready
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(5)

# Connect to index
index = pc.Index(PINECONE_INDEX_NAME)

# ----------------- MAIN FUNCTION -----------------


async def load_vectorstore(uploaded_files, role: str, doc_id: str):
    """
    Save uploaded PDFs, create text chunks, embed with HuggingFace model,
    and upsert into Pinecone with metadata.
    """
    # FREE 1024-dim embedding model (no Google / no paid API)
    embed_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en"
    )

    for file in uploaded_files:
        # 1. Save uploaded file locally
        save_path = Path(UPLOADS_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())

        # 2. Load PDF
        loader = PyPDFLoader(str(save_path))
        documents = loader.load()

        # 3. Split into smaller chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        chunks = splitter.split_documents(documents)

        texts = [chunk.page_content for chunk in chunks]
        ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": file.filename,
                "doc_id": doc_id,
                "role": role,
                "page": chunk.metadata.get("page", 0),
                "text": chunk.page_content,
            }
            for i, chunk in enumerate(chunks)
        ]

        print(f"Embedding {len(texts)} chunks with HuggingFace model (BAAI/bge-large-en)...")
        # Offload sync embedding call to a thread so it doesn't block event loop
        embeddings = await asyncio.to_thread(embed_model.embed_documents, texts)

        print("Uploading vectors to Pinecone...")
        vectors = [
            {"id": ids[i], "values": embeddings[i], "metadata": metadatas[i]}
            for i in range(len(embeddings))
        ]

        with tqdm(total=len(vectors), desc="Upserting to Pinecone") as progress:
            index.upsert(vectors=vectors)
            progress.update(len(vectors))

        print(f"Upload complete for {file.filename}")
