#!/usr/bin/env python3
"""
Convert CSV and TXT files into text embeddings (ChromaDB).

Supports:
- CSV files (each row → text chunk + metadata)
- Plain text files (.txt) (each file → one document + metadata)

Usage (from this folder or repo root):
  python csv_and_txt_to_embeddings.py                       # uses S3_BUCKET / EMBEDDINGS_INPUT_S3_PREFIX from config.py
  python csv_and_txt_to_embeddings.py --s3-prefix data/v2/  # override the S3 folder
  python csv_and_txt_to_embeddings.py --s3-bucket other-bucket --s3-prefix some/folder/
  python csv_and_txt_to_embeddings.py --help
"""

import argparse
import io
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from config import S3_BUCKET, EMBEDDINGS_INPUT_S3_PREFIX
from s3_utils import list_keys_with_suffix, fetch_object_bytes


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "input"  # kept for reference; no longer used by main()
VECTOR_DB_DIR = SCRIPT_DIR / "vector_db"

COLLECTION_NAME = "csv_txt_embeddings"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 128
ENCODINGS = ("utf-8", "latin-1", "iso-8859-1", "cp1252", "utf-8-sig")


def load_csv(path: Path) -> pd.DataFrame:
    """Local-disk CSV loader. Kept for reference / offline use; main() now uses load_csv_from_s3()."""
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read CSV: {path}")


def load_text_file(path: Path) -> str:
    """Local-disk text loader. Kept for reference / offline use; main() now uses load_text_from_s3()."""
    for enc in ENCODINGS:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read text file: {path}")


def load_csv_from_s3(bucket: str, key: str) -> pd.DataFrame:
    """Download a CSV object from S3 and parse it, trying multiple encodings."""
    raw_bytes = fetch_object_bytes(bucket, key)
    for enc in ENCODINGS:
        try:
            return pd.read_csv(io.BytesIO(raw_bytes), encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read CSV: s3://{bucket}/{key}")


def load_text_from_s3(bucket: str, key: str) -> str:
    """Download a .txt object from S3 and decode it, trying multiple encodings."""
    raw_bytes = fetch_object_bytes(bucket, key)
    for enc in ENCODINGS:
        try:
            return raw_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode text file: s3://{bucket}/{key}")


def _sanitize_meta_val(val: Any) -> Any:
    if isinstance(val, bool):
        return val
    try:
        f = float(val)
        return int(f) if f == int(f) else f
    except (ValueError, TypeError):
        pass
    s = str(val).strip()[:500]
    return s if s else None


def row_to_text_and_metadata(row: pd.Series, columns: list) -> Tuple[str, Dict[str, Any]]:
    parts = []
    metadata: Dict[str, Any] = {}
    for col in columns:
        val = row.get(col)
        if pd.isna(val) or val == "":
            continue
        parts.append(f"{col}: {val}")
        key = col.lower().replace(" ", "_").replace("-", "_")[:50]
        clean = _sanitize_meta_val(val)
        if clean is not None:
            metadata[key] = clean
    return " | ".join(parts), metadata


def build_csv_chunks(df: pd.DataFrame, source_name: str) -> List[Dict[str, Any]]:
    columns = list(df.columns)
    chunks: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        text, meta = row_to_text_and_metadata(row, columns)
        if not text.strip():
            continue
        meta["row_index"] = int(idx)
        meta["_source_file"] = source_name
        meta["source_type"] = "csv_row"
        chunks.append({"text": text, "metadata": meta})
    return chunks


def build_txt_chunks(path: Path, base_input: Path) -> List[Dict[str, Any]]:
    """Local-disk version. Kept for reference / offline use; main() now uses build_txt_chunks_from_s3()."""
    try:
        content = load_text_file(path)
    except Exception as e:
        print(f"⚠️  Skip text file {path.name}: {e}")
        return []
    content = content.strip()
    if not content:
        return []
    try:
        rel = str(path.relative_to(base_input))
    except ValueError:
        rel = path.name
    meta: Dict[str, Any] = {
        "filename": path.name,
        "path": str(path),
        "relative_path": rel,
        "source_type": "text_file",
    }
    return [{"text": content, "metadata": meta}]


def build_txt_chunks_from_s3(bucket: str, key: str, base_prefix: str) -> List[Dict[str, Any]]:
    """S3 version of build_txt_chunks(). `key` is the full S3 key; `base_prefix` is the
    scanned folder, used to compute a relative_path the same way path.relative_to() did."""
    try:
        content = load_text_from_s3(bucket, key)
    except Exception as e:
        print(f"⚠️  Skip text file {key}: {e}")
        return []
    content = content.strip()
    if not content:
        return []
    rel = key[len(base_prefix):].lstrip("/") if key.startswith(base_prefix) else key
    filename = key.split("/")[-1]
    meta: Dict[str, Any] = {
        "filename": filename,
        "path": f"s3://{bucket}/{key}",
        "relative_path": rel,
        "source_type": "text_file",
    }
    return [{"text": content, "metadata": meta}]


def collect_files_recursive(root: Path, suffix: str) -> List[Path]:
    """Local-disk version. Kept for reference / offline use; main() now uses list_keys_with_suffix()."""
    if root.is_file():
        return [root] if root.suffix.lower() == suffix else []
    return sorted(p for p in root.rglob(f"*{suffix}") if p.is_file())


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert an S3 folder (CSV + .txt) into text embeddings (ChromaDB)."
    )
    ap.add_argument(
        "--s3-bucket",
        default=S3_BUCKET,
        help=f"S3 bucket to read CSV/TXT files from (default: {S3_BUCKET})",
    )
    ap.add_argument(
        "--s3-prefix",
        default=EMBEDDINGS_INPUT_S3_PREFIX,
        help=f"S3 folder/prefix to scan recursively (default: {EMBEDDINGS_INPUT_S3_PREFIX})",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for vector DB (default: ./output)",
    )
    ap.add_argument(
        "--collection-name",
        default=COLLECTION_NAME,
        help=f"ChromaDB collection name (default: {COLLECTION_NAME})",
    )
    ap.add_argument(
        "--embedding-model",
        default=EMBEDDING_MODEL,
        help=f"SentenceTransformer model (default: {EMBEDDING_MODEL})",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for encoding (default: {BATCH_SIZE})",
    )
    args = ap.parse_args()

    bucket = args.s3_bucket
    prefix = args.s3_prefix

    vector_db_path = args.output_dir / "vector_db"
    vector_db_path.mkdir(parents=True, exist_ok=True)
    print(f"📂 Vector DB: {vector_db_path.absolute()}")
    print(f"☁️  Reading input from s3://{bucket}/{prefix}")

    # Collect CSV and TXT keys (recursively, so nested folders like book_text_files_Mar3/ are included)
    csv_keys = list_keys_with_suffix(bucket, prefix, ".csv")
    txt_keys = list_keys_with_suffix(bucket, prefix, ".txt")

    if not csv_keys and not txt_keys:
        print(f"   No CSV or .txt files found under s3://{bucket}/{prefix}")
        print("   Nothing to embed.")
        return

    print(f"📄 CSV files: {len(csv_keys)}")
    print(f"📝 Text files: {len(txt_keys)}")

    all_texts: List[str] = []
    all_metas: List[Dict[str, Any]] = []

    # CSV → row-level chunks
    for csv_key in csv_keys:
        try:
            df = load_csv_from_s3(bucket, csv_key)
        except Exception as e:
            print(f"⚠️  Skip CSV {csv_key}: {e}")
            continue
        source_name = csv_key.split("/")[-1]
        chunks = build_csv_chunks(df, source_name)
        for c in chunks:
            all_texts.append(c["text"])
            all_metas.append(c["metadata"])

    # TXT → one doc per file
    for txt_key in txt_keys:
        chunks = build_txt_chunks_from_s3(bucket, txt_key, prefix)
        for c in chunks:
            all_texts.append(c["text"])
            all_metas.append(c["metadata"])

    if not all_texts:
        print("   No non-empty text content to embed.")
        return

    print(f"🧾 Total text chunks: {len(all_texts)}")
    print(f"🤖 Loading embedding model: {args.embedding_model}")
    model = SentenceTransformer(args.embedding_model)

    try:
        client = chromadb.PersistentClient(
            path=str(vector_db_path),
            settings=Settings(anonymized_telemetry=False),
        )
    except Exception:
        client = chromadb.Client(
            Settings(persist_directory=str(vector_db_path), anonymized_telemetry=False)
        )

    collection = client.get_or_create_collection(
        name=args.collection_name,
        metadata={"source": "csv_txt", "model": args.embedding_model},
    )

    ids = [f"doc_{i}" for i in range(len(all_texts))]

    for i in range(0, len(all_texts), args.batch_size):
        batch_texts = all_texts[i : i + args.batch_size]
        batch_metas = all_metas[i : i + args.batch_size]
        batch_ids = ids[i : i + args.batch_size]
        batch_emb = model.encode(batch_texts).tolist()
        try:
            collection.add(
                ids=batch_ids,
                embeddings=batch_emb,
                documents=batch_texts,
                metadatas=batch_metas,
            )
        except TypeError:
            collection.add(
                embeddings=batch_emb,
                documents=batch_texts,
                metadatas=batch_metas,
                ids=batch_ids,
            )

    print(f"✅ Done. Collection '{args.collection_name}': {collection.count()} documents.")
    print(f"   Output: {vector_db_path.absolute()}")


if __name__ == "__main__":
    main()
