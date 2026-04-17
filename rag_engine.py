import re
import chromadb
from sentence_transformers import SentenceTransformer

class RAGEngine:
    def __init__(self, db_path="./chroma_db"):
        # Initialize persistent chromadb local storage implicitly handling connections
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        
        # Pull down the memory-efficient 22MB dimension encoder
        print("[RAG] Booting SentenceTransformer local weights (miniLM)...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Instantiate Collection 
        self.collection = self.chroma_client.get_or_create_collection(name="board_notes")

    def _clean_ocr(self, raw_text):
        """Standardizes messy OCR extraction prior to embedding generation"""
        if not raw_text: return ""
        # Preserve formulas via explicit filtering 
        cleaned = re.sub(r'[^a-zA-Z0-9\s.,=+\-()*/^<>]', ' ', raw_text)
        return re.sub(r'\s+', ' ', cleaned).strip()

    def sync_db_to_vector_store(self, sqlite_rows):
        """Upserts SQL dictionary rows as independent unchunked documents"""
        ids, documents, embeddings, metadatas = [], [], [], []
        
        for row in sqlite_rows:
            text = self._clean_ocr(row.get("raw_text", ""))
            
            # Skip invalid fragmented captures measuring < 5 length
            if len(text) < 5:
                continue
                
            doc_id = f"capture_{row['id']}"
            
            ids.append(doc_id)
            documents.append(text)
            embeddings.append(self.encoder.encode(text).tolist())
            
            # Preserve hardware contexts 
            metadatas.append({
                "timestamp": row.get("timestamp", ""),
                "image_path": row.get("image_path", "")
            })
            
        if ids:
            self.collection.upsert(
                ids=ids, embeddings=embeddings,
                documents=documents, metadatas=metadatas
            )
            print(f"[RAG] Successfully validated {len(ids)} board documents to VectorStore.")

    def retrieve_context(self, query, top_k=3):
        """Finds closest semantic matching board coordinates natively"""
        query_vec = self.encoder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k
        )
        
        matches = []
        if results and "documents" in results and results["documents"]:
            for i in range(len(results["documents"][0])):
                matches.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i]
                })
        return matches
