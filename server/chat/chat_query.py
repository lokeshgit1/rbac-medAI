import os
import asyncio
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load environment variables from .env
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")  # should be "chatbot"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not set in .env")
if not PINECONE_INDEX_NAME:
    raise ValueError("PINECONE_INDEX_NAME is not set in .env")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in .env")

# ---------------- PINECONE CLIENT ---------------- #

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# ---------------- EMBEDDINGS MODEL ---------------- #

# Must match what you used in docs/vectorstore.py (BAAI/bge-large-en → 1024-dim)
embed_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en"
)

# ---------------- LLM (GROQ) ---------------- #

llm = ChatGroq(
    temperature=0.3,
    model_name="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY
)



# ---------------- PROMPT TEMPLATE ---------------- #






prompt = PromptTemplate.from_template(
    """
You are a helpful healthcare assistant. Answer the following question
based ONLY on the provided context.

If the context is not sufficient to answer confidently, say so clearly.

Question:
{question}

Context:
{context}

If relevant, mention which document(s) the information came from.
"""
)

rag_chain = prompt | llm

# ---------------- MAIN RAG FUNCTION ---------------- #


async def answer_query(query: str, user_role: str):
    """
    Given a user question and their role (e.g. 'doctor', 'admin', 'patient'),
    this function:
      1. Embeds the query with HuggingFace (BAAI/bge-large-en)
      2. Queries Pinecone index for top_k similar chunks
      3. Filters chunks by `role` in metadata
      4. Sends the combined context to Groq Llama3
      5. Returns the LLM answer + sources
    """

    # 1. Embed the query
    query_embedding = await asyncio.to_thread(embed_model.embed_query, query)

    # 2. Query Pinecone (include metadata to get text + role + source)
    results = await asyncio.to_thread(
        index.query,
        vector=query_embedding,
        top_k=10,
        include_metadata=True,
        
    )

    filtered_chunks = []
    sources = set()

    # 3. Filter by role and build context
    for match in results.get("matches", []):
        metadata = match.get("metadata", {}) or {}

        # Optional: filter by role
        if user_role and metadata.get("role") != user_role:
            continue

        # Text should have been stored in metadata during upsert
        text = metadata.get("text", "")
        if text:
            filtered_chunks.append(text)

        source = metadata.get("source")
        if source:
            sources.add(source)

    if not filtered_chunks:
        return {
            "answer": "No relevant information found for your role in the indexed documents.",
            "sources": [],
        }

    # 4. Build final context text
    docs_text = "\n\n---\n\n".join(filtered_chunks)

    # 5. Call LLM via LangChain chain
    final_answer = await asyncio.to_thread(
        rag_chain.invoke, {"question": query, "context": docs_text}
    )

    # 6. Return answer + sources
    return {
        "answer": getattr(final_answer, "content", str(final_answer)),
        "sources": list(sources),
    }
