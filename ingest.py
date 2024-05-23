import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_PATH = 'source_documents/'
DB_FAISS_PATH = 'vectorstore/db_faiss'
INGEST_THREADS = 4  # Adjust this based on your system's capabilities

def load_single_document(file_path: str):
    # Loads a single document from a file path
    try:
        loader = PyPDFLoader(file_path)
        return loader.load()[0]
    except Exception as ex:
        print(f"Error loading document {file_path}: {ex}")
        return None

def load_document_batch(filepaths):
    # create a thread pool
    with ThreadPoolExecutor(len(filepaths)) as executor:
        # load files
        futures = [executor.submit(load_single_document, filepath) for filepath in filepaths]
        # collect data
        return [future.result() for future in futures]

def load_documents(source_dir: str):
    # Loads all documents from the source documents directory, including nested folders
    paths = []
    for root, _, files in os.walk(source_dir):
        for file_name in files:
            if file_name.endswith('.pdf'):
                paths.append(os.path.join(root, file_name))

    # Have at least one worker and at most INGEST_THREADS workers
    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    chunksize = max(1, len(paths) // n_workers)
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        # split the load operations into chunks
        for i in range(0, len(paths), chunksize):
            # select a chunk of filenames
            filepaths = paths[i : (i + chunksize)]
            # submit the task
            futures.append(executor.submit(load_document_batch, filepaths))
        
        # process all results
        for future in as_completed(futures):
            docs.extend(future.result())

    return [doc for doc in docs if doc is not None]

def create_vector_db():
    documents = load_documents(DATA_PATH)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800,  # Increase chunk size
                                                   chunk_overlap=100)  # Adjust overlap
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()
