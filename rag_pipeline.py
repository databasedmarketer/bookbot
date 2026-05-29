"""
RAG Pipeline that combines retrieval and generation for book summaries queries
"""

from typing import List, Dict, Any, Optional
import time
from vector_store import BookVectorStore
from llm_client import BookLLMClient
from data_processor import BookSummariesProcessor
from config import TOP_K_RESULTS

import re
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from config import VECTOR_DB_DIR, COLLECTION_NAME_EN, COLLECTION_NAME_FR, COLLECTION_NAME_BOOKS, EMBEDDING_MODEL, TOP_K_RESULTS

MAX_PROMPT_WORDS = 50  # Single place to change the limit across all entry points


class BookRAGPipeline:
    def __init__(self):
        self.vector_store = BookVectorStore()
        self.llm_client = BookLLMClient()
        self.processor = BookSummariesProcessor()
        self.is_initialized = False

    def initialize(self) -> bool:
        """Initialize the complete RAG pipeline"""
        print("🚀 Initializing RAG Pipeline...")

        try:
            # Connect to pre-built Chroma embeddings (csv_and_txt_to_embeddings.py output)
            print(f"📂 Connecting to pre-built vector DB: {VECTOR_DB_DIR.absolute()}")
            if not VECTOR_DB_DIR.exists():
                print(f"❌ Vector DB not found at {VECTOR_DB_DIR}")
                return False
            try:
                chroma_client = chromadb.PersistentClient(
                    path=str(VECTOR_DB_DIR),
                    settings=Settings(anonymized_telemetry=False),
                )
            except Exception:
                chroma_client = chromadb.Client(
                    Settings(persist_directory=str(VECTOR_DB_DIR), anonymized_telemetry=False)
                )
            self._chroma_collection = chroma_client.get_collection(name=COLLECTION_NAME_BOOKS)
            print(f"✅ Collection '{COLLECTION_NAME_BOOKS}': {self._chroma_collection.count()} documents")
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
            print(f"🤖 Embedding model loaded: {EMBEDDING_MODEL}")

            # Initialize LLM client
            print("🤖 Setting up LLM client...")
            try:
                if not self.llm_client.setup():
                    print("❌ Failed to setup LLM client")
                    return False
            except Exception as e:
                print(f"❌ Exception during LLM client setup: {e}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                return False

            self.is_initialized = True
            print("✅ RAG Pipeline initialized successfully!")
            return True

        except Exception as e:
            print(f"❌ Error initializing RAG pipeline: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return False

    def _normalize(self, text: str) -> str:
        """Normalize text for fuzzy title matching."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def _match_txt(self, book_title: str, txt_docs: list) -> dict | None:
        """Fuzzy-match a book title to a TXT doc by its filename stem."""
        needle = self._normalize(book_title)
        for doc in txt_docs:
            stem = self._normalize(
                doc["metadata"].get("filename", "")
                    .replace(".txt", "")
                    .replace("_", " ")
                    .replace("-", " ")
            )
            if needle == stem or needle in stem or stem in needle:
                return doc
        return None

    def _retrieve_and_merge(self, question: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Query pre-built Chroma collection, split into CSV rows vs TXT docs,
        then merge each CSV row with its matching interview transcript.
        """
        embedding = self._encoder.encode(question).tolist()
        results = self._chroma_collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        csv_items, txt_docs = [], []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            item = {
                "document":        doc,
                "metadata":        meta,
                "relevance_score": round(1 - dist, 4),
            }
            if meta.get("source_type") == "text_file":
                txt_docs.append(item)
            else:
                if not meta.get("book_title"):
                    for part in doc.split("|"):
                        part = part.strip()
                        if part.lower().startswith("book_title:"):
                            meta["book_title"] = part.split(":", 1)[1].strip()
                            break
                csv_items.append(item)

        # Fallback: if no TXT docs surfaced in top-n, search by title directly
        if csv_items and not txt_docs:
            seen = set()
            for item in csv_items:
                title = item["metadata"].get("book_title", "")
                if not title or title in seen:
                    continue
                seen.add(title)
                t_emb = self._encoder.encode(title).tolist()
                t_res = self._chroma_collection.query(
                    query_embeddings=[t_emb],
                    n_results=3,
                    include=["documents", "metadatas", "distances"],
                )
                for d, m, dist in zip(
                    t_res["documents"][0],
                    t_res["metadatas"][0],
                    t_res["distances"][0],
                ):
                    if m.get("source_type") == "text_file":
                        txt_docs.append({
                            "document":        d,
                            "metadata":        m,
                            "relevance_score": round(1 - dist, 4),
                        })

        # Merge each CSV row with its matching TXT transcript
        enriched = []
        for item in csv_items:
            book_title = item["metadata"].get("book_title", "")
            matched = self._match_txt(book_title, txt_docs)
            item = dict(item)
            if matched:
                item["document"] = (
                    item["document"]
                    + "\n\n--- Interview Transcript ---\n"
                    + matched["document"]
                )
                item["metadata"]["has_transcript"] = True
            else:
                item["metadata"]["has_transcript"] = False
            enriched.append(item)

        return enriched

    def query(
        self,
        question: str,
        max_results: int = TOP_K_RESULTS,
        language: str = "en",
        temperature: float = 0.1,
        excluded_books: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Process a query through the complete RAG pipeline.

        Parameters
        ----------
        excluded_books:
            Book titles to exclude from recommendations.  Only set when the
            user has explicitly asked to skip particular books.
        """
        if not self.is_initialized:
            raise ValueError("Pipeline not initialized. Call initialize() first.")

        word_count = len(question.split())
        if word_count > MAX_PROMPT_WORDS:
            return {
                'question': question,
                'answer': (
                    f"⚠️ Your question is too long ({word_count} words). "
                    f"Please keep it under {MAX_PROMPT_WORDS} words."
                ),
                'retrieved_documents': [],
                'processing_time': 0,
                'error': None
            }

        start_time = time.time()

        try:
            print(f"🔍 Searching for relevant documents...")
            retrieved_docs = self._retrieve_and_merge(question, n_results=max_results)

            if not retrieved_docs:
                return {
                    'question': question,
                    'answer': "I couldn't find any relevant book information for your question.",
                    'retrieved_documents': [],
                    'processing_time': time.time() - start_time,
                    'error': None
                }

            print(f"💭 Generating answer...")
            effective_language = language or "en"
            answer = self.llm_client.answer_book_questions(
                question,
                retrieved_docs,
                language=effective_language,
                temperature=temperature,
                excluded_books=excluded_books or [],
            )

            return {
                'question': question,
                'answer': answer,
                'retrieved_documents': retrieved_docs,
                'processing_time': time.time() - start_time,
                'error': None
            }

        except Exception as e:
            return {
                'question': question,
                'answer': f"Error processing your question: {str(e)}",
                'retrieved_documents': [],
                'processing_time': time.time() - start_time,
                'error': str(e)
            }

    def query_with_filters(
        self,
        question: str,
        book_title: Optional[str] = None,
        book_category: Optional[str] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        max_results: int = TOP_K_RESULTS,
        language: str = "en",
        temperature: float = 0.1,
        excluded_books: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Query with specific metadata filters."""
        if not self.is_initialized:
            raise ValueError("Pipeline not initialized. Call initialize() first.")

        word_count = len(question.split())
        if word_count > MAX_PROMPT_WORDS:
            return {
                'question': question,
                'answer': (
                    f"⚠️ Your question is too long ({word_count} words). "
                    f"Please keep it under {MAX_PROMPT_WORDS} words."
                ),
                'retrieved_documents': [],
                'processing_time': 0,
                'error': None
            }

        start_time = time.time()

        try:
            retrieved_docs = self._retrieve_and_merge(question, n_results=max_results)

            if not retrieved_docs:
                return {
                    'question': question,
                    'answer': "No books found matching your criteria.",
                    'retrieved_documents': [],
                    'processing_time': time.time() - start_time,
                    'error': None
                }

            effective_language = language or "en"
            answer = self.llm_client.answer_book_questions(
                question,
                retrieved_docs,
                language=effective_language,
                temperature=temperature,
                excluded_books=excluded_books or [],
            )

            return {
                'question': question,
                'answer': answer,
                'retrieved_documents': retrieved_docs,
                'processing_time': time.time() - start_time,
                'error': None
            }

        except Exception as e:
            return {
                'question': question,
                'answer': f"Error processing your question: {str(e)}",
                'retrieved_documents': [],
                'processing_time': time.time() - start_time,
                'error': str(e)
            }

    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the RAG system"""
        if not self.is_initialized:
            return {"status": "not_initialized"}

        vector_info = self.vector_store.get_collection_info()
        data_summary = self.processor.get_data_summary()

        return {
            "status": "initialized",
            "vector_database": vector_info,
            "data_summary": data_summary,
            "llm_model": self.llm_client.model,
            "embedding_model": (
                self.vector_store.embedding_model.get_sentence_embedding_dimension()
                if self.vector_store.embedding_model
                else None
            )
        }

    def get_sample_queries(self) -> List[str]:
        return self.processor.get_sample_queries()

    def clear_database(self):
        if self.is_initialized:
            self.vector_store.clear_collection()
            print("🗑️ Vector database cleared")


class BookQueryProcessor:
    """Helper class for processing and understanding book queries"""

    @staticmethod
    def extract_query_intent(query: str) -> Dict[str, Any]:
        query_lower = query.lower()
        intent = {'type': 'general', 'entities': [], 'filters': {}}

        if any(w in query_lower for w in ['summary', 'summarize', 'about', 'what']):
            intent['type'] = 'summary_inquiry'
        if any(w in query_lower for w in ['find', 'search', 'show', 'list']):
            intent['type'] = 'search'
        if any(w in query_lower for w in ['compare', 'difference', 'vs']):
            intent['type'] = 'comparison'

        for cat in ['marketing', 'business', 'economics']:
            if cat in query_lower:
                intent['entities'].append(cat)

        return intent


if __name__ == "__main__":
    pipeline = BookRAGPipeline()

    if pipeline.initialize():
        print("\n" + "=" * 50)
        print("RAG Pipeline Test")
        print("=" * 50)

        test_queries = [
            "What books are about marketing?",
            "Find books on business communication",
            "What are the main themes in the classical marketing book?"
        ]

        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            result = pipeline.query(query)
            print(f"⏱️ Processing time: {result['processing_time']:.2f}s")
            print(f"📄 Retrieved {len(result['retrieved_documents'])} documents")
            print(f"💬 Answer: {result['answer'][:200]}...")
            print("-" * 50)
    else:
        print("❌ Failed to initialize RAG pipeline")