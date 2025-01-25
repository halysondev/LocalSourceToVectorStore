import os
import sys
import chardet
from typing import Iterator
from pathlib import Path
import concurrent.futures  # For optional parallel processing
from tqdm import tqdm      # For the progress bar

#############################################
# LangChain / Qdrant / etc. Imports
#############################################
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_community.docstore.document import Document
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.language.language_parser import LanguageParser
from langchain_community.document_loaders.blob_loaders import FileSystemBlobLoader, Blob, BlobLoader

# We'll define our own minimal version of a list-based blob loader:
class MyListBlobLoader(BlobLoader):
    """A basic list-based blob loader if ListBlobLoader isn't available."""
    def __init__(self, blobs: list[Blob]):
        super().__init__()
        self._blobs = blobs

    def yield_blobs(self) -> Iterator[Blob]:
        yield from self._blobs


# Git import (only needed if you want to clone a repo)
from git import Repo

# Embeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Vector stores
from langchain_qdrant import QdrantVectorStore
from langchain_community.vectorstores.chroma import Chroma

# Qdrant client
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance


#############################################
# Initial Configurations
#############################################
FROM_GIT = False
GIT_URL = ""
repo_path = "Server178"

USE_OPENAI = False  # False => HuggingFaceEmbeddings
os.environ["OPENAI_API_KEY"] = "sk-xxx-YourKeyHere"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

CHUNK_SIZE = 500
CHUNK_OVERLAP = 200
PARSER_THRESHOLD = 1000

VECTOR_CHROMA = False
VECTOR_QDRANT = True

collection_name = "server178"
qdrant_url = "http://127.0.0.1:6333"

BATCH_SIZE = 250
MAX_WORKERS = 1  # or more if you want concurrency
DEVICE_TYPE = "cuda"

# Excluded suffixes we always skip
EXCLUDED_SUFFIXES = {
    ".data", ".conf", ".exe", ".lib", ".dll", ".so", ".bin",
    ".o", ".a", ".out"
}

# If an extension is in this set, we skip the binary heuristic check
TEXT_EXT_WHITELIST = {".hpp"}

ENCODINGS_TO_TRY = ["gb18030", "gb2312", "utf-8", "utf-16", "latin-1"]


#############################################
# (Optional) Clone a Git Repo
#############################################
if FROM_GIT:
    Repo.clone_from(GIT_URL, to_path=repo_path)


#############################################
# Binary-Check Helper
#############################################
def is_likely_binary(file_path: str, sample_size=4096, non_ascii_threshold=0.3) -> bool:
    """Return True if the file looks binary based on a quick heuristic."""
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(sample_size)
        if b"\x00" in chunk:
            return True
        non_ascii = sum(b > 127 for b in chunk)
        ratio = non_ascii / len(chunk) if chunk else 0
        return ratio > non_ascii_threshold
    except:
        return True


#############################################
# Decode function with multiple encodings
#############################################
def try_decode(data: bytes, encodings=ENCODINGS_TO_TRY):
    for enc in encodings:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    import chardet
    result = chardet.detect(data)
    encoding_guessed = result["encoding"]
    confidence = result["confidence"]
    if encoding_guessed and confidence > 0.5:
        try:
            return data.decode(encoding_guessed, errors="replace")
        except UnicodeDecodeError:
            pass
    return None


#############################################
# Custom Parser
#############################################
class MultiEncodingLanguageParser(LanguageParser):
    def _guess_language(self, blob: Blob) -> Language:
        ext = Path(blob.source).suffix.lower()
        if ext == ".py":
            return Language.PYTHON
        elif ext in [".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx", ".hh"]:
            return Language.CPP
        elif ext == ".js":
            return Language.JS
        elif ext == ".java":
            return Language.JAVA
        elif ext == ".lua":
            return Language.LUA
        elif ext == ".php":
            return Language.PHP
        elif ext == ".rb":
            return Language.RUBY
        elif ext == ".rs":
            return Language.RUST
        elif ext == ".swift":
            return Language.SWIFT
        elif ext == ".ts":
            return Language.TS
        elif ext == ".html":
            return Language.HTML
        elif ext == ".perl" or ext == ".pl":
            return Language.PERL
        else:
            return Language.MARKDOWN

    def _parse_text(self, text: str, blob: Blob, language: Language) -> Iterator[Document]:
        yield Document(
            page_content=text,
            metadata={"source": blob.source, "language": language.name}
        )

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        data = blob.as_bytes()
        code_str = try_decode(data)
        if code_str is None:
            print(f"Skipping {blob.source} – unable to decode.")
            return
        language = self.language or self._guess_language(blob)
        yield from self._parse_text(code_str, blob, language)


#############################################
# Load & Filter Files
#############################################
blob_loader = FileSystemBlobLoader(repo_path, glob="**/*")
all_blobs = list(blob_loader.yield_blobs())

filtered_blobs = []
for blob in all_blobs:
    ext = Path(blob.path).suffix.lower()

    # 1) Exclude by known binary extension
    if ext in EXCLUDED_SUFFIXES:
        print(f"Skipping {blob.path} (excluded extension).")
        continue

    # 2) Bypass the binary check if extension is in text whitelist
    if ext not in TEXT_EXT_WHITELIST:
        if is_likely_binary(blob.path):
            print(f"Skipping {blob.path} (likely binary).")
            continue

    filtered_blobs.append(blob)

print(f"Total files found: {len(all_blobs)}")
print(f"Excluded files: {len(all_blobs) - len(filtered_blobs)}")
print(f"Files to parse: {len(filtered_blobs)}")

# Create a "list" loader for the filtered Blobs
list_blob_loader = MyListBlobLoader(filtered_blobs)

loader = GenericLoader(
    blob_loader=list_blob_loader,
    blob_parser=MultiEncodingLanguageParser(parser_threshold=PARSER_THRESHOLD),
)
documents = loader.load()
print(f"Loaded {len(documents)} documents.")


#############################################
# Split Documents into Chunks
#############################################
all_chunks = []
for doc in documents:
    lang_str = doc.metadata.get("language", "MARKDOWN").upper()
    if lang_str in Language.__members__:
        lang_enum = Language[lang_str]
    else:
        lang_enum = Language.MARKDOWN

    try:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=lang_enum,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
    except ValueError:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

    chunks = splitter.split_documents([doc])
    all_chunks.extend(chunks)

print(f"Original docs: {len(documents)}")
print(f"Total chunks after splitting: {len(all_chunks)}")


#############################################
# Choose Which Embedding to Use
#############################################
if USE_OPENAI:
    print("Using OpenAIEmbeddings ... (text-embedding-ada-002 => 1536 dims)")
    embeddings = OpenAIEmbeddings(disallowed_special=())
    embedding_dimension = 1536
else:
    print(f"Using HuggingFaceEmbeddings ... ({EMBEDDING_MODEL_NAME} => {EMBEDDING_DIMENSION} dims)")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": DEVICE_TYPE}
    )
    embedding_dimension = EMBEDDING_DIMENSION


#############################################
# Initialize Vector Store (Qdrant or Chroma)
#############################################
vectordb = None

if VECTOR_QDRANT:
    qdrant_client = QdrantClient(url=qdrant_url, prefer_grpc=False)
    try:
        qdrant_client.get_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' already exists; deleting for fresh start.")
        qdrant_client.delete_collection(collection_name=collection_name)
    except:
        pass

    print(f"Creating Qdrant collection '{collection_name}' with size={embedding_dimension}.")
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE),
    )
    vectordb = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    print("Using Qdrant as vector store...")

elif VECTOR_CHROMA:
    vectordb = Chroma(
        collection_name=collection_name,
        embeddings=embeddings,
        persist_directory="./data"
    )
    print("Using Chroma as vector store...")

if vectordb is None:
    raise ValueError("No vector store selected! Please set VECTOR_QDRANT or VECTOR_CHROMA to True.")


#############################################
# Insert Documents in Parallel Batches with tqdm
#############################################
def process_batch(idx, docs_batch):
    """Embed and add a single batch to vectordb. Return (idx, count)."""
    vectordb.add_documents(docs_batch)
    return (idx, len(docs_batch))

from concurrent.futures import ThreadPoolExecutor

# Instead of as_completed, we store the futures by index and wait in ascending order
batches = []
for i in range(0, len(all_chunks), BATCH_SIZE):
    batch_docs = all_chunks[i : i + BATCH_SIZE]
    batches.append((i // BATCH_SIZE, batch_docs))

futures = {}
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor, tqdm(
    total=len(all_chunks),
    desc="Inserting documents into Vector Store (in order)",
) as pbar:
    # Submit tasks keyed by batch index
    for batch_idx, docs_batch in batches:
        future = executor.submit(process_batch, batch_idx, docs_batch)
        futures[batch_idx] = future

    # Now process results in ascending batch_idx order
    for batch_idx, docs_batch in batches:
        inserted_idx, inserted_count = futures[batch_idx].result()
        # We know inserted_idx == batch_idx, but let's be explicit
        pbar.update(inserted_count)

if VECTOR_CHROMA:
    vectordb.persist()

print("Done creating the vector store!")
