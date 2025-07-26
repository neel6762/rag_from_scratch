"""
This module is responsible for loading the data from the data directory.
It supports the following file types: PDF, TXT, CSV.
"""

import os
import logging
import pandas as pd
from pypdf import PdfReader

logging.basicConfig(level=logging.ERROR, format='%(levelname)s [%(filename)s]: %(message)s')
logger = logging.getLogger(__name__)

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

    def peek_data(self, data: dict[str, str]):

        for key, value in data.items():
            print(f"File: {key}")
            print(f"\nContent: {value[:100]}")
            print("-" * 100)
            print("\n")
    

    def load_files(self) -> dict[str, any]:

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
                elif file.endswith('.txt') or file.endswith('.md'):
                    data[file] = self._load_text(os.path.join(self.data_dir, file))
                elif file.endswith('.csv'):
                    data[file] = self._load_csv(os.path.join(self.data_dir, file))     
                else:
                    logger.warning(f"Unsupported file type {file}")
                    continue

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
    