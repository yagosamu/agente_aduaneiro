import json
import logging
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class NCMTabelaIndexer:
    """Indexa a Tabela NCM oficial em FAISS para busca semantica."""

    def __init__(
        self,
        ncm_path: str = "data/ncm",
        vectorstore_path: str = "data/vectorstore",
    ):
        self.ncm_path = Path(ncm_path)
        self.vectorstore_path = Path(vectorstore_path)

    # ------------------------------------------------------------------
    # Leitura do JSON
    # ------------------------------------------------------------------

    def carregar_json(self) -> list[dict]:
        """Le o primeiro JSON encontrado em ncm_path e retorna a lista de Nomenclaturas."""
        json_files = sorted(self.ncm_path.glob("Tabela_NCM*.json"))
        if not json_files:
            raise FileNotFoundError(
                f"Nenhum arquivo Tabela_NCM*.json encontrado em {self.ncm_path}"
            )

        json_file = json_files[0]
        logger.info("Carregando tabela NCM: %s", json_file.name)

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        nomenclaturas = data.get("Nomenclaturas", [])
        logger.info("Total de entradas na tabela: %d", len(nomenclaturas))
        return nomenclaturas

    # ------------------------------------------------------------------
    # Limpeza de descricao
    # ------------------------------------------------------------------

    @staticmethod
    def _limpar_descricao(desc: str) -> str:
        """Remove tags HTML e marcadores hierarquicos (- e --) do inicio."""
        texto = re.sub(r"<[^>]+>", "", desc)
        texto = re.sub(r"^-+\s*", "", texto.strip())
        return texto.strip()

    # ------------------------------------------------------------------
    # Hierarquia
    # ------------------------------------------------------------------

    @staticmethod
    def _codigo_normalizado(codigo: str) -> str:
        """Remove pontos do codigo NCM."""
        return codigo.replace(".", "")

    def _resolver_hierarquia(self, codigo: str, lookup: dict[str, str]) -> str:
        """Monta texto enriquecido com hierarquia para um NCM de 8 digitos.

        Exemplo para 7318.15.00:
            NCM 7318.15.00: Outros parafusos e pinos...
            Capitulo 73: Obras de ferro fundido, ferro ou aco.
            Posicao 73.18: Parafusos, pinos ou pernos, roscados...
            Subposicao 7318.1: Artigos roscados
        """
        digits = self._codigo_normalizado(codigo)
        desc_ncm = self._limpar_descricao(lookup.get(codigo, ""))

        partes = [f"NCM {codigo}: {desc_ncm}"]

        # Capitulo (2 digitos)
        cap_code = digits[:2]
        if cap_code in lookup:
            partes.append(f"Capitulo {cap_code}: {self._limpar_descricao(lookup[cap_code])}")

        # Posicao (4 digitos -> XX.XX)
        pos_code = f"{digits[:2]}.{digits[2:4]}"
        if pos_code in lookup:
            partes.append(f"Posicao {pos_code}: {self._limpar_descricao(lookup[pos_code])}")

        # Subposicao de 1o nivel (5 digitos)
        sub5_code = f"{digits[:4]}.{digits[4]}"
        if sub5_code in lookup:
            partes.append(f"Subposicao {sub5_code}: {self._limpar_descricao(lookup[sub5_code])}")

        # Subposicao de 2o nivel (6 digitos -> XXXX.XX)
        sub6_code = f"{digits[:4]}.{digits[4:6]}"
        if sub6_code in lookup:
            partes.append(f"Subposicao {sub6_code}: {self._limpar_descricao(lookup[sub6_code])}")

        return "\n".join(partes)

    # ------------------------------------------------------------------
    # Construcao de chunks
    # ------------------------------------------------------------------

    def construir_chunks(self, nomenclaturas: list[dict]) -> tuple[list[str], list[str]]:
        """Constroi chunks enriquecidos para NCMs de 8 digitos.

        Returns:
            (lista_chunks, lista_codigos) para os NCMs completos.
        """
        # Lookup: codigo -> descricao (todos os niveis)
        lookup: dict[str, str] = {}
        for item in nomenclaturas:
            lookup[item["Codigo"]] = item["Descricao"]

        chunks: list[str] = []
        codigos: list[str] = []

        for item in nomenclaturas:
            digits = self._codigo_normalizado(item["Codigo"])
            if len(digits) != 8:
                continue

            texto = self._resolver_hierarquia(item["Codigo"], lookup)
            chunks.append(texto)
            codigos.append(item["Codigo"])

        logger.info("Chunks gerados: %d NCMs de 8 digitos", len(chunks))
        return chunks, codigos

    # ------------------------------------------------------------------
    # Build completo
    # ------------------------------------------------------------------

    def build(self, model: SentenceTransformer | None = None) -> int:
        """Pipeline completo: JSON -> chunks -> embeddings -> FAISS.

        Returns:
            Numero de NCMs indexados.
        """
        nomenclaturas = self.carregar_json()
        chunks, codigos = self.construir_chunks(nomenclaturas)

        if not chunks:
            raise ValueError("Nenhum NCM de 8 digitos encontrado para indexar.")

        if model is None:
            logger.info("Carregando modelo de embeddings: %s", EMBEDDING_MODEL)
            model = SentenceTransformer(EMBEDDING_MODEL)

        logger.info("Gerando embeddings para %d NCMs...", len(chunks))
        embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
        embeddings = np.array(embeddings, dtype="float32")

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        logger.info("Indice FAISS NCM criado: %d vetores (dim=%d)", index.ntotal, dimension)

        # Salvar
        self.vectorstore_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(self.vectorstore_path / "ncm_tabela.faiss"))
        np.save(str(self.vectorstore_path / "ncm_tabela_chunks.npy"), np.array(chunks, dtype=object))
        np.save(str(self.vectorstore_path / "ncm_tabela_codigos.npy"), np.array(codigos, dtype=object))

        logger.info("Indice NCM salvo em %s", self.vectorstore_path)
        return len(chunks)


# ------------------------------------------------------------------
# Uso standalone
# ------------------------------------------------------------------

if __name__ == "__main__":
    indexer = NCMTabelaIndexer()
    total = indexer.build()
    print(f"Tabela NCM indexada: {total} codigos")
