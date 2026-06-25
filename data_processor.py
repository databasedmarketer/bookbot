"""
Data processing module for book summaries CSV
"""

import pandas as pd
import numpy as np
import io
from typing import List, Dict, Any
from config import BOOKS_CSV_FILE, CHUNK_SIZE, CHUNK_OVERLAP, S3_BUCKET, BOOKS_CSV_S3_KEY, TXT_FILES_S3_PREFIX
from s3_utils import fetch_csv_bytes, fetch_txt_files
import re

class BookSummariesProcessor:
    def __init__(self):
        self.df = None
        self.processed_chunks = []
    
    def load_data(self) -> pd.DataFrame:
        """Load and clean the book summaries CSV data"""
        print(f"Loading data from s3://{S3_BUCKET}/{BOOKS_CSV_S3_KEY}")

        # Download the CSV bytes from S3 (replaces the old local-disk read)
        raw_bytes = fetch_csv_bytes(S3_BUCKET, BOOKS_CSV_S3_KEY)

        # Try different encodings for CSV files (some may have special characters)
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-8-sig']
        self.df = None
        last_error = None
        
        for encoding in encodings:
            try:
                self.df = pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding)
                print(f"✅ Successfully loaded CSV with {encoding} encoding")
                break
            except UnicodeDecodeError as e:
                last_error = e
                continue
            except Exception as e:
                raise IOError(
                    f"Failed to read CSV file s3://{S3_BUCKET}/{BOOKS_CSV_S3_KEY}: {e}\n"
                    f"Please check if the file is valid CSV format."
                )
        
        if self.df is None:
            raise IOError(
                f"Failed to read CSV file s3://{S3_BUCKET}/{BOOKS_CSV_S3_KEY} with any encoding (tried: {', '.join(encodings)})\n"
                f"Last error: {last_error}\n"
                f"Please check if the file is valid CSV format."
            )

        # Clean the data - drop rows with missing book title
        self.df = self.df.dropna(subset=['book_title'])
        
        # Fill empty descriptions with a default value
        self.df['book_description'] = self.df['book_description'].fillna('No description available')
        self.df['book_blob_notes'] = self.df['book_blob_notes'].fillna('')
        self.df['book_excerpt2'] = self.df['book_excerpt2'].fillna('')
        self.df['book_excerpt3'] = self.df['book_excerpt3'].fillna('')
        self.df['book_category'] = self.df['book_category'].fillna('Uncategorized')
        self.df['book_subtitle'] = self.df['book_subtitle'].fillna('')
        
        print(f"Loaded {len(self.df)} books")
        return self.df

    def load_txt_transcripts(self) -> List[Dict[str, str]]:
        """
        Fetch interview-transcript .txt files from the S3 data folder
        (replaces reading them from a local txt/ subfolder). Intended for
        use by the embeddings-building script.
        Returns a list of {'filename': ..., 'content': ...} dicts.
        """
        print(f"Loading TXT transcripts from s3://{S3_BUCKET}/{TXT_FILES_S3_PREFIX}")
        txt_files = fetch_txt_files(S3_BUCKET, TXT_FILES_S3_PREFIX)
        print(f"Loaded {len(txt_files)} txt files")
        return [{"filename": fn, "content": content} for fn, content in txt_files]
    
    def create_chunks(self) -> List[Dict[str, Any]]:
        """Create chunks from the book summaries data for better retrieval"""
        if self.df is None:
            self.load_data()
        
        chunks = []
        
        for idx, row in self.df.iterrows():
            # Combine all text content from the book
            full_text = self._create_chunk_text(row)
            
            # Split long content into smaller chunks if needed
            if len(full_text) > CHUNK_SIZE:
                sub_chunks = self._split_long_text(full_text, row)
                chunks.extend(sub_chunks)
            else:
                chunks.append({
                    'text': full_text,
                    'metadata': {
                        'book_id': int(row['id']) if pd.notna(row['id']) else idx,
                        'book_title': str(row['book_title']),
                        'book_subtitle': str(row['book_subtitle']) if pd.notna(row['book_subtitle']) else '',
                        'book_category': str(row['book_category']),
                        'book_pubyear': int(row['book_pubyear']) if pd.notna(row['book_pubyear']) else None,
                        'row_index': idx
                    }
                })
        
        self.processed_chunks = chunks
        print(f"Created {len(chunks)} chunks from book summaries")
        return chunks
    
    def _create_chunk_text(self, row: pd.Series) -> str:
        """Create a comprehensive text representation of a book summary"""
        text_parts = []
        
        # Title and subtitle
        title = str(row['book_title'])
        if pd.notna(row['book_subtitle']) and str(row['book_subtitle']).strip():
            title += f": {row['book_subtitle']}"
        text_parts.append(f"Title: {title}")
        
        # Category
        if pd.notna(row['book_category']):
            text_parts.append(f"Category: {row['book_category']}")
        
        # Publication year
        if pd.notna(row['book_pubyear']):
            text_parts.append(f"Publication Year: {int(row['book_pubyear'])}")
        
        # Description
        if pd.notna(row['book_description']) and str(row['book_description']).strip():
            text_parts.append(f"Description: {row['book_description']}")
        
        # Detailed notes/summary (this is the main content)
        if pd.notna(row['book_blob_notes']) and str(row['book_blob_notes']).strip():
            text_parts.append(f"Summary: {row['book_blob_notes']}")
        
        # Excerpts
        if pd.notna(row['book_excerpt2']) and str(row['book_excerpt2']).strip():
            text_parts.append(f"Excerpt: {row['book_excerpt2']}")
        
        if pd.notna(row['book_excerpt3']) and str(row['book_excerpt3']).strip():
            text_parts.append(f"Excerpt: {row['book_excerpt3']}")
        
        return "\n\n".join(text_parts)
    
    def _split_long_text(self, text: str, row: pd.Series) -> List[Dict[str, Any]]:
        """Split long text into smaller chunks with overlap"""
        # Split by sentences first for better chunking
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > CHUNK_SIZE and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        'book_id': int(row['id']) if pd.notna(row['id']) else row.name,
                        'book_title': str(row['book_title']),
                        'book_subtitle': str(row['book_subtitle']) if pd.notna(row['book_subtitle']) else '',
                        'book_category': str(row['book_category']),
                        'book_pubyear': int(row['book_pubyear']) if pd.notna(row['book_pubyear']) else None,
                        'row_index': row.name,
                        'chunk_index': len(chunks)
                    }
                })
                # Keep overlap
                overlap_words = int(CHUNK_OVERLAP * 0.5)  # Keep some words for overlap
                current_chunk = current_chunk[-overlap_words:] if overlap_words < len(current_chunk) else []
                current_length = sum(len(word) for word in current_chunk)
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    'book_id': int(row['id']) if pd.notna(row['id']) else row.name,
                    'book_title': str(row['book_title']),
                    'book_subtitle': str(row['book_subtitle']) if pd.notna(row['book_subtitle']) else '',
                    'book_category': str(row['book_category']),
                    'book_pubyear': int(row['book_pubyear']) if pd.notna(row['book_pubyear']) else None,
                    'row_index': row.name,
                    'chunk_index': len(chunks)
                }
            })
        
        return chunks
    
    def get_sample_queries(self) -> List[str]:
        """Generate sample queries for testing"""
        return [
            "What books are about marketing?",
            "Find books on business communication",
            "What are the main themes in the classical marketing book?",
            "Show me books published in 2025",
            "What books discuss persuasion and rhetoric?",
            "Find books about storytelling",
            "What are the key insights from business books?",
            "Show me books in the Business & Economics category"
        ]
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the data"""
        if self.df is None:
            self.load_data()
        
        return {
            'total_books': len(self.df),
            'unique_categories': self.df['book_category'].nunique() if 'book_category' in self.df.columns else 0,
            'publication_years': {
                'min': int(self.df['book_pubyear'].min()) if pd.notna(self.df['book_pubyear'].min()) else None,
                'max': int(self.df['book_pubyear'].max()) if pd.notna(self.df['book_pubyear'].max()) else None,
            },
            'top_categories': self.df['book_category'].value_counts().head().to_dict() if 'book_category' in self.df.columns else {}
        }

if __name__ == "__main__":
    processor = BookSummariesProcessor()
    processor.load_data()
    chunks = processor.create_chunks()
    summary = processor.get_data_summary()
    
    print("\nData Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print(f"\nSample chunk:")
    print(chunks[0]['text'][:200] + "...")
