"""
Data loading and preprocessing module for folktales
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import logging
from tqdm import tqdm
import requests
import re
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class FolktaleDataLoader:
    """Load and preprocess folktale data from various sources"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize data loader with configuration"""
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.data_config = self.config['data']
    
    def load_western_tales(self) -> pd.DataFrame:
        """
        Load Western folktales with ATU labels from CSV
        
        Returns:
            DataFrame with columns: [text, atu_label, source, category]
        """
        logger.info(f"Loading Western folktales from {self.data_config['western_tales_path']}")
        
        csv_path = Path(self.data_config['western_tales_path'])
        if not csv_path.exists():
            raise FileNotFoundError(f"Western tales CSV not found at {csv_path}")
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} Western folktales")
        
        return df
    
    def load_asian_tales(self) -> pd.DataFrame:
        """
        Load Asian/Southeast Asian folktales from JSON datasets
        
        Expected JSON structure: List of tales with 'text' field and optional metadata
        
        Returns:
            DataFrame with columns: [text, source_file, region, title, ...]
        """
        logger.info(f"Scanning Asian tales from {self.data_config['asian_tales_dir']}")
        
        asian_dir = Path(self.data_config['asian_tales_dir'])
        asian_dir.mkdir(parents=True, exist_ok=True)
        
        tales = []
        for json_file in asian_dir.glob('**/*.json'):
            try:
                logger.info(f"Loading {json_file}")
                
                # Load JSON file
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = pd.read_json(f)
                
                # Handle different JSON structures
                if isinstance(data, pd.DataFrame):
                    # JSON is already a DataFrame
                    tales_data = data
                elif isinstance(data, list):
                    # JSON is a list of dictionaries
                    tales_data = pd.DataFrame(data)
                else:
                    logger.warning(f"Unexpected JSON structure in {json_file}")
                    continue
                
                # Extract region from filename (e.g., "china_china_fables_dataset.json" -> "china")
                filename = json_file.stem
                region = filename.split('_')[0] if '_' in filename else filename
                
                # Process each tale
                for idx, tale in tales_data.iterrows():
                    tale_dict = {
                        'source_file': str(json_file),
                        'region': region,
                        'title': f"{region}_tale_{idx}",
                    }
                    
                    # Extract text field (try common variations)
                    text = None
                    for text_field in ['text', 'story', 'content', 'tale', 'fable']:
                        if text_field in tale and pd.notna(tale[text_field]):
                            text = str(tale[text_field])
                            break
                    
                    if text is None:
                        logger.warning(f"No text field found for tale {idx} in {json_file}")
                        continue
                    
                    tale_dict['text'] = text
                    
                    # Add any other metadata fields
                    for col in tales_data.columns:
                        if col not in ['text', 'story', 'content', 'tale', 'fable']:
                            tale_dict[col] = tale[col]
                    
                    tales.append(tale_dict)
                    
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
                logger.warning(f"Error details: {str(e)}")
        
        df = pd.DataFrame(tales)
        logger.info(f"Loaded {len(df)} Asian folktales from {len(list(asian_dir.glob('**/*.json')))} JSON files")
        
        return df
    
    def preprocess_text(self, text: str) -> str:
        """
        Basic text preprocessing
        
        Args:
            text: Raw folktale text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove common markup/artifacts
        text = text.replace('***', '')
        text = re.sub(r'_+', '', text)
        
        return text
    
    def split_long_texts(self, df: pd.DataFrame, max_length: int = 'auto') -> pd.DataFrame:
        """
        Split very long texts into multiple entries (optional for large texts)
        
        Args:
            df: DataFrame with 'text' column
            max_length: Maximum text length; 'auto' uses median length
            
        Returns:
            DataFrame with potentially more rows
        """
        if max_length == 'auto':
            max_length = int(df['text'].str.len().median() * 1.5)
            logger.info(f"Auto-setting max_length to {max_length} chars")
        
        new_rows = []
        for idx, row in df.iterrows():
            text = row['text']
            if len(text) > max_length:
                # Simple sentence-based splitting
                sentences = text.split('. ')
                chunks = []
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < max_length:
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + ". "
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                for chunk in chunks:
                    new_row = row.copy()
                    new_row['text'] = chunk
                    new_rows.append(new_row)
            else:
                new_rows.append(row)
        
        return pd.DataFrame(new_rows).reset_index(drop=True)
    
    def save_processed_data(self, df: pd.DataFrame, name: str):
        """Save processed data for reproducibility"""
        processed_dir = Path(self.data_config['processed_dir'])
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = processed_dir / f"{name}.csv"
        df.to_csv(filepath, index=False)
        logger.info(f"Saved processed data to {filepath}")


class EmbeddingCache:
    """Cache embeddings to avoid recomputation"""
    
    def __init__(self, cache_dir: str = "data/processed/embeddings/"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_path(self, hash_key: str) -> Path:
        """Get path for cached embeddings"""
        return self.cache_dir / f"{hash_key}.npy"
    
    def save_embeddings(self, embeddings: np.ndarray, hash_key: str):
        """Save embeddings to cache"""
        np.save(self.get_cache_path(hash_key), embeddings)
    
    def load_embeddings(self, hash_key: str) -> np.ndarray:
        """Load embeddings from cache"""
        path = self.get_cache_path(hash_key)
        if path.exists():
            return np.load(path)
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = FolktaleDataLoader()
    
    # Example usage
    # western_df = loader.load_western_tales()
    # asian_df = loader.load_asian_tales()
