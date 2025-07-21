import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s [%(filename)s]: %(message)s')

logger = logging.getLogger(__name__)

class Indexer:
    ...

if __name__ == "__main__":
    logger.info("Indexing data...")
        
        