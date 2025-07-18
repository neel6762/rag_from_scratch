import os
import logging
import pandas as pd
from pypdf import PdfReader

logging.basicConfig(level=logging.INFO, format='%(levelname)s [%(filename)s]: %(message)s')
logger = logging.getLogger("loader")

class Loader:

    def __init__(
            self, 
            data_dir: str,
            exclude_file_types: list[str] = None,
            exclude_file_names: list[str] = None
            ):
        
        self.data_dir = data_dir
        self.exclude_file_types = exclude_file_types
        self.exclude_file_names = exclude_file_names

    
    def load_files(self) -> dict[str, str]:

        data = {}

        for file in os.listdir(self.data_dir):

            if not os.path.isfile(os.path.join(self.data_dir, file)):
                logger.info(f"Skipping file - Not a file {file}")
                continue

            if self.exclude_file_types and file.endswith(tuple(self.exclude_file_types)):
                logger.info(f"Skipping file - Excluded file type {file}")
                continue
            
            if self.exclude_file_names and file in self.exclude_file_names:
                logger.info(f"Skipping file - Excluded file name {file}")
                continue

            else:
                logger.info(f"Reading file - {file}")
                
                if file.endswith('.pdf'):
                    data[file] = self._load_pdf(os.path.join(self.data_dir, file))
                elif file.endswith('.txt'):
                    data[file] = self._load_text(os.path.join(self.data_dir, file))

        return data
    
    def _load_pdf(self, file: str) -> str:

        try:
            reader = PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
            
            return text
        
        except Exception as e:
            logger.error(f"Error loading PDF file {file} - {e}")
            return None

    def _load_text(self, file: str) -> str:
        
        try:
            with open(file, 'r') as f:
                return f.read()
            
        except Exception as e:
            logger.error(f"Error loading text file {file} - {e}")
            return None

    def _load_csv(self, file: str) -> pd.DataFrame:
        
        try:
            return pd.read_csv(file)
        
        except Exception as e:
            logger.error(f"Error loading CSV file {file} - {e}")
            return None
    

if __name__ == '__main__':
    
    loader = Loader(data_dir='data', exclude_file_types=['.csv'])
    files = loader.load_files()