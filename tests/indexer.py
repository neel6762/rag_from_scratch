import pytest
from rag import Vectorizer, Loader

@pytest.fixture
def vectorizer_instance():
    """Fixture for Vectorizer instance."""
    loader = Loader(data_dir="data", exclude_file_types=["csv"])
    data = loader.load_files()
    
    # Using small chunk size for testing purposes
    return Vectorizer(data=data, chunk_size=100, chunk_overlap=20)


def test_vectorize_docs(vectorizer_instance):
    """Test splitting documents into chunks."""
    vectorizer_instance.vectorize_docs()
    assert vectorizer_instance.db_client.collection.get() is not None, "Chunks should not be None"
    assert len(vectorizer_instance.db_client.collection.get()["documents"]) > 0, "Chunks should have at least one file"
    assert isinstance(vectorizer_instance.db_client.collection.get()["documents"][0], str), "Chunks should be a string"

@pytest.mark.parametrize("file_name", ["history_of_cricket.md"])
def test_get_file_chunks(vectorizer_instance, file_name):
    """Test retrieving chunks for a specific file."""
    vectorizer_instance.vectorize_docs()
    chunks = vectorizer_instance.get_file_chunks(file_name)
    assert chunks is not None
    assert isinstance(chunks, list)
    assert len(chunks) > 0

def test_get_file_chunks_not_found(vectorizer_instance):
    """Test retrieving chunks for a non-existent file."""
    with pytest.raises(ValueError, match="File non_existent_file.txt not found in data"):
        vectorizer_instance.get_file_chunks("non_existent_file.txt")

def test_chunking_logic():
    """Test the chunking logic with a sample text."""
    file_name = "sample.txt"
    text = ("word " * 200).strip()
    data = {file_name: text}
    vectorizer = Vectorizer(data, chunk_size=50, chunk_overlap=10)
    vectorizer.vectorize_docs()
    chunks = vectorizer.get_file_chunks(file_name)
    
    # With 200 words, chunk_size=50, and overlap=10, the step is 40.
    # The number of chunks should be ceil((200 - 10) / (50 - 10)) = 5
    assert len(chunks) == 5

    # Verify the content of the chunks and the overlap
    words = text.split(" ")
    
    # Chunk 1: words[0:50]
    assert chunks[0] == " ".join(words[0:50])
    
    # Chunk 2: words[40:90]
    assert chunks[1] == " ".join(words[40:90])
    
    # Chunk 5: words[160:200]
    assert chunks[4] == " ".join(words[160:200])

    # Check overlap between chunk 1 and 2
    chunk1_words = chunks[0].split()
    chunk2_words = chunks[1].split()
    assert chunk1_words[-10:] == chunk2_words[:10]
