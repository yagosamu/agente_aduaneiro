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

        # Indice NCM (tabela oficial)
        self.ncm_chunks: list[str] = []
        self.ncm_codigos: list[str] = []
        self.ncm_index: faiss.IndexFlatL2 | None = None

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
    # Tabela NCM — load / search
    # ------------------------------------------------------------------

    def load_ncm_vectorstore(self) -> None:
        """Carrega indice FAISS da tabela NCM oficial."""
        index_file = self.vectorstore_path / "ncm_tabela.faiss"
        chunks_file = self.vectorstore_path / "ncm_tabela_chunks.npy"
        codigos_file = self.vectorstore_path / "ncm_tabela_codigos.npy"

        if not index_file.exists() or not chunks_file.exists() or not codigos_file.exists():
            raise FileNotFoundError(
                f"Indice NCM nao encontrado em {self.vectorstore_path}. "
                "Execute NCMTabelaIndexer.build() primeiro."
            )

        self.ncm_index = faiss.read_index(str(index_file))
        self.ncm_chunks = list(np.load(str(chunks_file), allow_pickle=True))
        self.ncm_codigos = list(np.load(str(codigos_file), allow_pickle=True))

        if self.model is None:
            logger.info("Carregando modelo de embeddings: %s", EMBEDDING_MODEL)
            self.model = SentenceTransformer(EMBEDDING_MODEL)

        logger.info(
            "Indice NCM carregado: %d vetores, %d codigos",
            self.ncm_index.ntotal,
            len(self.ncm_codigos),
        )

    def search_ncm(self, query: str, k: int = 3) -> list[dict]:
        """Busca os k NCMs mais relevantes na tabela oficial.

        Retorna lista de dicts com 'text', 'score', 'ncm_codigo' e 'source'.
        """
        if self.ncm_index is None or not self.ncm_chunks:
            return []

        query_embedding = self.create_embeddings([query])
        distances, indices = self.ncm_index.search(query_embedding, min(k, len(self.ncm_chunks)))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.ncm_chunks):
                results.append({
                    "text": self.ncm_chunks[idx],
                    "score": float(dist),
                    "ncm_codigo": self.ncm_codigos[idx],
                    "source": "ncm_tabela",
                })

        logger.info("Busca NCM por '%s': %d resultados", query[:60], len(results))
        return results

    def search_combined(self, query: str, k_nesh: int = 3, k_ncm: int = 3) -> list[dict]:
        """Busca em ambos os indices (NESH + NCM) e retorna resultados combinados.

        Cada resultado tem 'text', 'score' e 'source' ("nesh" ou "ncm_tabela").
        Resultados NCM tambem incluem 'ncm_codigo'.
        """
        results = []

        # Busca NESH
        if self.index is not None and self.chunks:
            nesh_results = self.search(query, k=k_nesh)
            for r in nesh_results:
                r["source"] = "nesh"
                results.append(r)

        # Busca NCM
        ncm_results = self.search_ncm(query, k=k_ncm)
        results.extend(ncm_results)

        # Ordena por score (menor distancia L2 = mais relevante)
        results.sort(key=lambda x: x["score"])

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
