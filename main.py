import logging
from rag import Loader, Vectorizer

# Configure logging to display INFO messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s [%(filename)s]: %(message)s')

if __name__ == "__main__":

    # ------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------
    loader = Loader(data_dir="data", exclude_file_types=["csv", "pdf"])
    data = loader.load_files()

    print(f"Loaded {len(data)} files")

    # ------------------------------------------------------------
    # VECTORIZE DATA
    # ------------------------------------------------------------
    file_name = "history_of_cricket.md"
    
    vectorizer = Vectorizer(data=data, chunk_size=500, chunk_overlap=150)
    vectorizer.vectorize_docs()

    print(vectorizer.get_file_chunks(file_name)[0])
    print(vectorizer.get_file_embeddings(file_name)[0])