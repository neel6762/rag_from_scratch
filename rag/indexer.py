import logging

logging.basicConfig(level=logging.ERROR, format='%(levelname)s [%(filename)s]: %(message)s')
logger = logging.getLogger(__name__)

class DocSplitter:
    
    def __init__(
            self, 
            data: dict[str, any], 
            chunk_size: int = 500, 
            chunk_overlap: int = 150,
        ) -> None:
        """
        Initialize the DocSplitter class.

        Args:
            data (dict[str, any]): The data to split.
            chunk_size (int): The size of each chunk.
            chunk_overlap (int): The overlap between chunks.
        """
        self.data = data
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = self._split_docs()

    def split_docs(self) -> dict[str, list[str]]:
        """
        Split the data into chunks.
        """
        return self._split_docs()
    
    
    def _split_docs(self) -> dict[str, list[str]]:
        """
        Split the data into chunks.

        Returns:
            dict[str, list[str]]: A dictionary of the split data.
        """

        content = {}   
        for file_name, file_content in self.data.items():
            chunks = []
            file_content = file_content.split(" ")
            for i in range(0, len(file_content), self.chunk_size - self.chunk_overlap):
                try:
                    chunk = file_content[i:i+self.chunk_size]
                except IndexError:
                    chunk = file_content[i:]
                except Exception as e:
                    logger.error(f"Error splitting {file_name}: {e}")
                    continue
                chunks.append(" ".join(chunk))
            content[file_name] = chunks

        return content


    def get_file_chunks(self, file_name: str) -> list[str]:
        """Get the chunks of a file.

        Args:
            file_name (str): The name of the file to get the chunks of.

        Returns:
            list[str]: The chunks of the file.
        """
        
        if file_name not in list(self.chunks.keys()):
            logger.error(f"File {file_name} not found in data")
            raise ValueError(f"File {file_name} not found in data")
        
        return self.chunks[file_name]