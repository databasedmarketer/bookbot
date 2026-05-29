"""
Vector store module using ChromaDB for book summaries and information. This module handles the creation of a vector database
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import os
from config import VECTOR_DB_DIR, COLLECTION_NAME_BOOKS, EMBEDDING_MODEL, TOP_K_RESULTS
from data_processor import BookSummariesProcessor

# ChromaDB version compatibility helper
def get_chromadb_version():
    """Get ChromaDB version"""
    try:
        return chromadb.__version__
    except:
        return "unknown"

class BookVectorStore:
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.processor = BookSummariesProcessor()

    def initialize(self):
        """Initialize ChromaDB client and collection"""
        print("Initializing ChromaDB...")
        
        # Initialize ChromaDB client (compatible with both old and new API)
        try:
            # Try new API first (chromadb >= 0.4.0)
            self.client = chromadb.PersistentClient(
                path=str(VECTOR_DB_DIR),
                settings=Settings(anonymized_telemetry=False)
            )
            self.use_new_api = True
        except (AttributeError, TypeError):
            # Fall back to old API (chromadb < 0.4.0)
            try:
                settings = Settings(
                    persist_directory=str(VECTOR_DB_DIR),
                    anonymized_telemetry=False
                )
                self.client = chromadb.Client(settings)
                self.use_new_api = False
            except Exception as e:
                # Last resort: try without settings
                self.client = chromadb.Client()
                self.use_new_api = False
        
        # Load embedding model
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        try:
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load embedding model '{EMBEDDING_MODEL}': {e}\n"
                f"This may be due to network issues or missing model files.\n"
                f"Please ensure you have internet connection or the model is cached locally."
            )
        
        # Get or create collection based on language
        self._ensure_collection()

    def _collection_name(self) -> str:
        return COLLECTION_NAME_BOOKS

    def _ensure_collection(self):
        name = self._collection_name()
        try:
            # Try get_or_create_collection first (works in both old and new API)
            if hasattr(self.client, 'get_or_create_collection'):
                try:
                    self.collection = self.client.get_or_create_collection(
                        name=name,
                        metadata={"description": f"Book summaries and information ({name})"}
                    )
                    print(f"✅ Collection '{name}' ready")
                except Exception as e:
                    # If get_or_create fails, try get/create separately
                    try:
                        self.collection = self.client.get_collection(name)
                        print(f"✅ Loaded existing collection: {name}")
                    except Exception:
                        self.collection = self.client.create_collection(
                            name=name,
                            metadata={"description": f"Book summaries and information ({name})"}
                        )
                        print(f"✅ Created new collection: {name}")
            else:
                # Fall back to try-except pattern for older API
                try:
                    self.collection = self.client.get_collection(name)
                    print(f"✅ Loaded existing collection: {name}")
                except Exception:
                    self.collection = self.client.create_collection(
                        name=name,
                        metadata={"description": f"Book summaries and information ({name})"}
                    )
                    print(f"✅ Created new collection: {name}")
        except Exception as e:
            print(f"❌ Error creating/loading collection: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise
    
    def populate_database(self):
        """Populate the vector database with book data"""
        print("Populating vector database...")
        
        # Get processed chunks
        chunks = self.processor.create_chunks()
        
        if not chunks:
            print("No chunks to process. Loading data first...")
            self.processor.load_data()
            chunks = self.processor.create_chunks()
        
        # Prepare data for ChromaDB
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedding_model.encode(texts).tolist()
        
        # Add to collection in batches to avoid API issues
        print("Adding documents to collection...")
        batch_size = 100  # Process in batches of 100
        total_added = 0
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            
            try:
                # Try different parameter orders for ChromaDB compatibility
                chroma_version = get_chromadb_version()
                if chroma_version.startswith("0.3"):
                    # ChromaDB 0.3.x workaround - use client's add method directly
                    try:
                        # Try using the client's add method with collection name
                        collection_name = self._collection_name()
                        if hasattr(self.client, 'add'):
                            self.client.add(
                                collection_name=collection_name,
                                ids=batch_ids,
                                embeddings=batch_embeddings,
                                documents=batch_texts,
                                metadatas=batch_metadatas
                            )
                        else:
                            # Fallback: try collection.add with minimal parameters
                            self.collection.add(
                                ids=batch_ids,
                                documents=batch_texts,
                                metadatas=batch_metadatas,
                                embeddings=batch_embeddings
                            )
                    except Exception as e1:
                        # Try alternative: add documents without embeddings (let ChromaDB generate)
                        try:
                            self.collection.add(
                                ids=batch_ids,
                                documents=batch_texts,
                                metadatas=batch_metadatas
                            )
                            print(f"  ⚠️ Batch {i//batch_size + 1} added without embeddings (ChromaDB will generate)")
                        except Exception as e2:
                            # Last resort: try upsert if available
                            if hasattr(self.collection, 'upsert'):
                                self.collection.upsert(
                                    ids=batch_ids,
                                    documents=batch_texts,
                                    metadatas=batch_metadatas,
                                    embeddings=batch_embeddings
                                )
                            else:
                                raise RuntimeError(
                                    f"ChromaDB 0.3.23 compatibility issue. Cannot add documents. "
                                    f"Error: {e1}. Alternative error: {e2}. "
                                    f"Please upgrade ChromaDB: pip install --upgrade --break-system-packages chromadb>=0.4.0"
                                )
                else:
                    # ChromaDB 0.4.x+ standard order
                    self.collection.add(
                        embeddings=batch_embeddings,
                        documents=batch_texts,
                        metadatas=batch_metadatas,
                        ids=batch_ids
                    )
                total_added += len(batch_texts)
                if (i // batch_size + 1) % 10 == 0:
                    print(f"  Added {total_added}/{len(texts)} documents...")
            except AttributeError as e:
                if "'Collection' object has no attribute '_client'" in str(e):
                    print(f"❌ ChromaDB version compatibility issue detected.")
                    print(f"   ChromaDB version: {get_chromadb_version()}")
                    print(f"   Attempting workaround...")
                    # Try workaround: add without embeddings
                    try:
                        self.collection.add(
                            ids=batch_ids,
                            documents=batch_texts,
                            metadatas=batch_metadatas
                        )
                        total_added += len(batch_texts)
                        print(f"  ⚠️ Batch {i//batch_size + 1} added without embeddings (workaround)")
                    except Exception as e2:
                        raise RuntimeError(
                            f"ChromaDB version {get_chromadb_version()} is incompatible. "
                            f"Please upgrade: pip install --upgrade --break-system-packages chromadb>=0.4.0"
                        )
                else:
                    raise
            except Exception as e:
                print(f"❌ Error adding batch {i//batch_size + 1}: {e}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                raise
        
        print(f"✅ Successfully added {total_added} documents to the collection")
    
    def search(self, query: str, n_results: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
        """Search for relevant book information using hybrid approach (exact + vector similarity)"""
        if not self.collection:
            raise ValueError("Collection not initialized. Call initialize() first.")
        
        # First, try exact title matching
        exact_results = self._search_exact_title(query)
        if exact_results:
            return exact_results
        
        # If no exact match, fall back to vector similarity search
        return self._search_vector_similarity(query, n_results)
    
    def _search_exact_title(self, query: str) -> List[Dict[str, Any]]:
        """Search for exact book title matches"""
        # Get all documents and search for exact title matches
        all_docs = self.collection.get()
        exact_matches = []
        
        if not all_docs or 'documents' not in all_docs or not all_docs['documents']:
            return []
        
        query_lower = query.lower()
        for i, doc in enumerate(all_docs['documents']):
            metadata = all_docs['metadatas'][i] if all_docs.get('metadatas') and i < len(all_docs['metadatas']) else {}
            book_title = metadata.get('book_title', '').lower()
            
            # Check if query contains book title or vice versa
            if book_title and (query_lower in book_title or book_title in query_lower):
                exact_matches.append({
                    'document': doc,
                    'metadata': metadata,
                    'distance': 0.0,
                    'relevance_score': 1.0  # Perfect match
                })
        
        return exact_matches[:5]  # Return top 5 exact matches
    
    def _search_vector_similarity(self, query: str, n_results: int) -> List[Dict[str, Any]]:
        """Search using vector similarity (original method)"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        formatted_results = []
        if not results or 'documents' not in results or not results['documents'] or len(results['documents'][0]) == 0:
            return formatted_results
        
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i] if results.get('metadatas') and results['metadatas'][0] else {},
                'distance': results['distances'][0][i] if results.get('distances') and results['distances'][0] else 0.0,
                'relevance_score': 1 - (results['distances'][0][i] if results.get('distances') and results['distances'][0] else 0.0)  # Convert distance to similarity
            })
        
        return formatted_results
    
    def search_by_metadata(self, 
                          book_title: Optional[str] = None,
                          book_category: Optional[str] = None,
                          min_year: Optional[int] = None,
                          max_year: Optional[int] = None,
                          n_results: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
        """Search by specific metadata filters"""
        if not self.collection:
            raise ValueError("Collection not initialized. Call initialize() first.")
        
        # Build where clause
        where_clause = {}
        if book_title:
            where_clause['book_title'] = book_title
        if book_category:
            where_clause['book_category'] = book_category
        if min_year is not None:
            where_clause['book_pubyear'] = {"$gte": min_year}
        if max_year is not None:
            if 'book_pubyear' in where_clause:
                where_clause['book_pubyear']['$lte'] = max_year
            else:
                where_clause['book_pubyear'] = {"$lte": max_year}
        
        # Query collection
        results = self.collection.query(
            where=where_clause if where_clause else None,
            n_results=n_results,
            include=['documents', 'metadatas']
        )
        
        # Format results
        formatted_results = []
        if not results or 'documents' not in results or not results['documents'] or len(results['documents'][0]) == 0:
            return formatted_results
        
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i] if results.get('metadatas') and results['metadatas'][0] else {}
            })
        
        return formatted_results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        if not self.collection:
            return {"error": "Collection not initialized"}
        
        # Handle both old and new chromadb API
        try:
            count = self.collection.count()
        except AttributeError:
            # Old API: use client's _count method with collection ID
            if hasattr(self.collection, 'id') and hasattr(self.client, '_count'):
                count = self.client._count(collection_id=self.collection.id)
            else:
                # Fallback: try to get count from get()
                try:
                    results = self.collection.get()
                    count = len(results['ids']) if results and 'ids' in results else 0
                except:
                    count = 0
        
        return {
            "name": self._collection_name(),
            "document_count": count,
            "embedding_model": EMBEDDING_MODEL
        }
    
    def clear_collection(self):
        """Clear all documents from the collection"""
        if self.collection:
            # Delete and recreate the collection
            name = self._collection_name()
            self.client.delete_collection(name)
            self.collection = self.client.create_collection(
                name=name,
                metadata={"description": f"Book summaries and information ({name})"}
            )
            print("Collection cleared")

if __name__ == "__main__":
    # Test the vector store
    vector_store = BookVectorStore()
    vector_store.initialize()
    
    # Check if collection is empty and populate if needed
    info = vector_store.get_collection_info()
    if info.get("document_count", 0) == 0:
        print("Collection is empty. Populating...")
        vector_store.populate_database()
    else:
        print(f"Collection has {info['document_count']} documents")
    
    # Test search
    print("\nTesting search...")
    results = vector_store.search("marketing", n_results=3)
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Relevance: {result['relevance_score']:.3f}")
        print(f"Book Title: {result['metadata'].get('book_title', 'N/A')}")
        print(f"Category: {result['metadata'].get('book_category', 'N/A')}")
        print(f"Description: {result['document'][:200]}...")
