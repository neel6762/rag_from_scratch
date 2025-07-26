import pytest
from rag import DocSplitter, Loader

@pytest.fixture
def doc_splitter_instance():
    """Fixture for DocSplitter instance."""
    loader = Loader(data_dir="data", exclude_file_types=["csv"])
    data = loader.load_files()
    # Using small chunk size for testing purposes
    return DocSplitter(data=data, chunk_size=100, chunk_overlap=20)


def test_split_docs(doc_splitter_instance):
    """Test splitting documents into chunks."""
    chunks = doc_splitter_instance.split_docs()
    assert chunks is not None, "Chunks should not be None"
    assert isinstance(chunks, dict), "Chunks should be a dictionary"
    assert len(chunks) > 0, "Chunks should have at least one file"
    
    a_file = list(chunks.keys())[0]
    assert len(chunks[a_file]) > 0
    assert isinstance(chunks[a_file][0], str)


def test_get_file_chunks(doc_splitter_instance):
    """Test retrieving chunks for a specific file."""
    file_name = list(doc_splitter_instance.data.keys())[0]
    chunks = doc_splitter_instance.get_file_chunks(file_name)
    assert chunks is not None
    assert isinstance(chunks, list)
    assert len(chunks) > 0


def test_get_file_chunks_not_found(doc_splitter_instance):
    """Test retrieving chunks for a non-existent file."""
    with pytest.raises(ValueError, match="File non_existent_file.txt not found in data"):
        doc_splitter_instance.get_file_chunks("non_existent_file.txt")


def test_chunking_logic():
    """Test the chunking logic with a sample text."""
    text = ("word " * 200).strip()
    data = {"sample.txt": text}
    splitter = DocSplitter(data, chunk_size=50, chunk_overlap=10)
    chunks = splitter.get_file_chunks("sample.txt")
    
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
