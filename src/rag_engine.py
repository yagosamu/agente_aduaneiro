import os
import logging
from pathlib import Path

import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import tiktoken

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


class RAGEngine:
    def __init__(
        self,
        docs_path: str = "data/nesh",
        vectorstore_path: str = "data/vectorstore",
    ):
        self.docs_path = Path(docs_path)
        self.vectorstore_path = Path(vectorstore_path)
        self.chunks: list[str] = []
        self.index: faiss.IndexFlatL2 | None = None
        self.model: SentenceTransformer | None = None
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    # ------------------------------------------------------------------
    # Documentos
    # ------------------------------------------------------------------

    def load_documents(self) -> list[str]:
        """Carrega todos os PDFs de `docs_path` e retorna lista de textos."""
        if not self.docs_path.exists():
            raise FileNotFoundError(f"Pasta nao encontrada: {self.docs_path}")

        pdf_files = sorted(self.docs_path.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"Nenhum PDF encontrado em {self.docs_path}")

        documents: list[str] = []
        for pdf_file in pdf_files:
            try:
                reader = PdfReader(str(pdf_file))
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                if text.strip():
                    documents.append(text)
                    logger.info("PDF carregado: %s (%d paginas)", pdf_file.name, len(reader.pages))
                else:
                    logger.warning("PDF sem texto extraivel: %s", pdf_file.name)
            except Exception:
                logger.exception("Erro ao ler PDF: %s", pdf_file.name)

        logger.info("Total de documentos carregados: %d", len(documents))
        return documents

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def _split_into_chunks(self, text: str) -> list[str]:
        """Divide texto em chunks de ~CHUNK_SIZE tokens com overlap."""
        tokens = self._tokenizer.encode(text)
        chunks: list[str] = []
        start = 0
        while start < len(tokens):
            end = start + CHUNK_SIZE
            chunk_tokens = tokens[start:end]
            chunk_text = self._tokenizer.decode(chunk_tokens)
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def create_embeddings(self, texts: list[str]) -> np.ndarray:
        """Gera embeddings para lista de textos usando sentence-transformers."""
        if self.model is None:
            logger.info("Carregando modelo de embeddings: %s", EMBEDDING_MODEL)
            self.model = SentenceTransformer(EMBEDDING_MODEL)

        logger.info("Gerando embeddings para %d chunks...", len(texts))
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        return np.array(embeddings, dtype="float32")

    # ------------------------------------------------------------------
    # Vector store
    # ------------------------------------------------------------------

    def build_vectorstore(self) -> None:
        """Pipeline completo: carrega PDFs -> chunking -> embeddings -> FAISS."""
        documents = self.load_documents()

        self.chunks = []
        for doc in documents:
            self.chunks.extend(self._split_into_chunks(doc))

        if not self.chunks:
            raise ValueError("Nenhum chunk gerado a partir dos documentos.")

        logger.info("Total de chunks gerados: %d", len(self.chunks))

        embeddings = self.create_embeddings(self.chunks)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        logger.info("Indice FAISS criado com %d vetores (dim=%d)", self.index.ntotal, dimension)

        self.save_vectorstore()

    def save_vectorstore(self) -> None:
        """Salva indice FAISS e chunks no disco."""
        if self.index is None:
            raise ValueError("Indice FAISS nao foi criado. Execute build_vectorstore() primeiro.")

        self.vectorstore_path.mkdir(parents=True, exist_ok=True)

        index_file = self.vectorstore_path / "nesh.faiss"
        chunks_file = self.vectorstore_path / "nesh_chunks.npy"

        faiss.write_index(self.index, str(index_file))
        np.save(str(chunks_file), np.array(self.chunks, dtype=object))

        logger.info("Vector store salvo em %s", self.vectorstore_path)

    def load_vectorstore(self) -> None:
        """Carrega indice FAISS e chunks do disco."""
        index_file = self.vectorstore_path / "nesh.faiss"
        chunks_file = self.vectorstore_path / "nesh_chunks.npy"

        if not index_file.exists() or not chunks_file.exists():
            raise FileNotFoundError(
                f"Vector store nao encontrado em {self.vectorstore_path}. "
                "Execute build_vectorstore() primeiro."
            )

        self.index = faiss.read_index(str(index_file))
        self.chunks = list(np.load(str(chunks_file), allow_pickle=True))

        if self.model is None:
            logger.info("Carregando modelo de embeddings: %s", EMBEDDING_MODEL)
            self.model = SentenceTransformer(EMBEDDING_MODEL)

        logger.info(
            "Vector store carregado: %d vetores, %d chunks",
            self.index.ntotal,
            len(self.chunks),
        )

    # ------------------------------------------------------------------
    # Busca
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 3) -> list[dict]:
        """Busca os k chunks mais relevantes para a query.

        Retorna lista de dicts com 'text' e 'score' (distancia L2).
        """
        if self.index is None or not self.chunks:
            raise ValueError(
                "Vector store nao carregado. "
                "Execute build_vectorstore() ou load_vectorstore() primeiro."
            )

        query_embedding = self.create_embeddings([query])
        distances, indices = self.index.search(query_embedding, min(k, len(self.chunks)))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                results.append({
                    "text": self.chunks[idx],
                    "score": float(dist),
                })

        logger.info("Busca por '%s': %d resultados", query[:60], len(results))
        return results


# ------------------------------------------------------------------
# Exemplo de uso
# ------------------------------------------------------------------

if __name__ == "__main__":
    engine = RAGEngine(
        docs_path="data/nesh",
        vectorstore_path="data/vectorstore",
    )

    # Tenta carregar vector store existente; se nao existir, cria um novo
    try:
        engine.load_vectorstore()
        print("Vector store carregado do disco.")
    except FileNotFoundError:
        print("Vector store nao encontrado. Criando novo...")
        engine.build_vectorstore()
        print("Vector store criado e salvo.")

    # Exemplo de busca
    query = "parafuso de aco inoxidavel classificacao"
    results = engine.search(query, k=3)

    print(f"\nResultados para: '{query}'\n")
    for i, result in enumerate(results, 1):
        print(f"--- Resultado {i} (score: {result['score']:.4f}) ---")
        print(result["text"][:300])
        print()
