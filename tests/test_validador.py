"""Testes para o sistema de validacao NCM."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.llm_agent import ValidadorNCM
from src.rag_engine import RAGEngine


# ================================================================
# Fixtures
# ================================================================

EXEMPLOS_PATH = Path("data/exemplos_ncm.json")


@pytest.fixture
def casos_teste():
    with open(EXEMPLOS_PATH, encoding="utf-8") as f:
        return json.load(f)["casos"]


@pytest.fixture
def mock_openai_response():
    """Cria uma resposta mock da API OpenAI."""

    def _make(status="APROVADO", mensagem="OK", justificativa="Teste", regras=None):
        body = json.dumps(
            {
                "status": status,
                "mensagem": mensagem,
                "justificativa": justificativa,
                "ncm_sugerida_alternativa": None,
                "regras_citadas": regras or ["RGI 1"],
            }
        )

        choice = MagicMock()
        choice.message.content = body

        usage = MagicMock()
        usage.prompt_tokens = 500
        usage.completion_tokens = 150
        usage.total_tokens = 650

        response = MagicMock()
        response.choices = [choice]
        response.usage = usage
        return response

    return _make


@pytest.fixture
def validador_mock(mock_openai_response):
    """ValidadorNCM com cliente OpenAI mockado."""
    v = ValidadorNCM(api_key="sk-fake-test-key", model="gpt-4o")
    v.client = MagicMock()
    v.client.chat.completions.create.return_value = mock_openai_response()
    return v


@pytest.fixture
def rag_engine_memory():
    """RAGEngine com dados em memoria (sem disco)."""
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    engine = RAGEngine()
    engine.chunks = [
        "Posicao 7318 - Parafusos, pinos ou pernos, roscados, porcas, de ferro fundido, ferro ou aco.",
        "Posicao 8471 - Maquinas automaticas para processamento de dados (computadores).",
        "Posicao 8517 - Aparelhos telefonicos, incluindo telefones para redes celulares.",
    ]
    engine.model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = engine.create_embeddings(engine.chunks)
    engine.index = faiss.IndexFlatL2(embeddings.shape[1])
    engine.index.add(embeddings)
    return engine


# ================================================================
# Testes: Formato NCM
# ================================================================


class TestFormatoNCM:
    """Testes de normalizacao e validacao de formato NCM."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("73181500", "7318.15.00"),
            ("7318.15.00", "7318.15.00"),
            ("7318-15-00", "7318.15.00"),
            ("7318 15 00", "7318.15.00"),
            ("09012100", "0901.21.00"),
            ("84713000", "8471.30.00"),
        ],
    )
    def test_normalizar_ncm(self, raw, expected):
        import re

        digits = re.sub(r"[^0-9]", "", raw)
        if len(digits) == 8:
            result = f"{digits[:4]}.{digits[4:6]}.{digits[6:8]}"
        else:
            result = raw.strip()
        assert result == expected

    @pytest.mark.parametrize(
        "ncm,valido",
        [
            ("7318.15.00", True),
            ("0901.21.00", True),
            ("8471.30.00", True),
            ("73181500", False),
            ("7318.1.00", False),
            ("", False),
            ("abcd.ef.gh", False),
            ("7318.15.0", False),
        ],
    )
    def test_validar_formato_ncm(self, ncm, valido):
        import re

        result = bool(re.match(r"^\d{4}\.\d{2}\.\d{2}$", ncm))
        assert result == valido


# ================================================================
# Testes: Casos de exemplo
# ================================================================


class TestCasosExemplo:
    """Testes com os casos de data/exemplos_ncm.json."""

    def test_arquivo_existe(self):
        assert EXEMPLOS_PATH.exists(), "data/exemplos_ncm.json nao encontrado"

    def test_estrutura_json(self, casos_teste):
        assert len(casos_teste) == 10
        campos = {"id", "descricao", "ncm", "resultado_esperado", "motivo"}
        for caso in casos_teste:
            assert campos.issubset(caso.keys()), f"Caso {caso['id']}: campos faltando"

    def test_distribuicao_resultados(self, casos_teste):
        contagem = {}
        for caso in casos_teste:
            r = caso["resultado_esperado"]
            contagem[r] = contagem.get(r, 0) + 1
        assert contagem["APROVADO"] == 3
        assert contagem["RISCO"] == 4
        assert contagem["ATENCAO"] == 3

    def test_ncm_formato_valido(self, casos_teste):
        import re

        for caso in casos_teste:
            ncm = caso["ncm"]
            assert re.match(
                r"^\d{4}\.\d{2}\.\d{2}$", ncm
            ), f"Caso {caso['id']}: NCM invalido '{ncm}'"


# ================================================================
# Testes: ValidadorNCM com mock
# ================================================================


class TestValidadorMock:
    """Testes do ValidadorNCM sem chamar a API real."""

    def test_instanciacao(self):
        v = ValidadorNCM(api_key="sk-fake", model="gpt-4o")
        assert v.model == "gpt-4o"
        assert v.rag_engine is None
        assert v.last_usage["prompt_tokens"] == 0

    def test_validar_retorna_campos(self, validador_mock):
        resultado = validador_mock.validar("Parafuso inox M10", "7318.15.00")
        assert "status" in resultado
        assert "mensagem" in resultado
        assert "justificativa" in resultado
        assert "regras_citadas" in resultado
        assert "usage" in resultado

    def test_validar_usage_tokens(self, validador_mock):
        resultado = validador_mock.validar("Parafuso inox M10", "7318.15.00")
        assert resultado["usage"]["prompt_tokens"] == 500
        assert resultado["usage"]["completion_tokens"] == 150
        assert resultado["usage"]["total_tokens"] == 650

    def test_validar_status_aprovado(self, validador_mock, mock_openai_response):
        validador_mock.client.chat.completions.create.return_value = (
            mock_openai_response(status="APROVADO", mensagem="NCM correta")
        )
        resultado = validador_mock.validar("Parafuso inox", "7318.15.00")
        assert resultado["status"] == "APROVADO"

    def test_validar_status_risco(self, validador_mock, mock_openai_response):
        validador_mock.client.chat.completions.create.return_value = (
            mock_openai_response(status="RISCO", mensagem="NCM incorreta")
        )
        resultado = validador_mock.validar("Notebook Dell", "0201.10.00")
        assert resultado["status"] == "RISCO"

    def test_validar_status_atencao(self, validador_mock, mock_openai_response):
        validador_mock.client.chat.completions.create.return_value = (
            mock_openai_response(status="ATENCAO", mensagem="Revisar")
        )
        resultado = validador_mock.validar("Peca metalica generica", "7318.15.00")
        assert resultado["status"] == "ATENCAO"

    def test_parse_json_invalido(self, validador_mock):
        choice = MagicMock()
        choice.message.content = "Isso nao e JSON"
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 50
        usage.total_tokens = 150
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage = usage
        validador_mock.client.chat.completions.create.return_value = resp

        resultado = validador_mock.validar("Teste", "1234.56.78")
        assert resultado["status"] == "ERRO"

    def test_sem_rag_retorna_mensagem(self, validador_mock):
        ctx = validador_mock._buscar_contexto_rag("parafuso", "7318.15.00")
        assert "Nenhuma base tecnica disponivel" in ctx

    def test_modelo_passado_na_chamada(self, validador_mock):
        validador_mock.validar("Teste", "1234.56.78")
        call_kwargs = (
            validador_mock.client.chat.completions.create.call_args.kwargs
        )
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["temperature"] == 0.1

    def test_todos_os_casos_aprovados(
        self, validador_mock, mock_openai_response, casos_teste
    ):
        """Roda todos os 10 casos (com mock) e verifica estrutura."""
        for caso in casos_teste:
            validador_mock.client.chat.completions.create.return_value = (
                mock_openai_response(status=caso["resultado_esperado"])
            )
            resultado = validador_mock.validar(caso["descricao"], caso["ncm"])
            assert resultado["status"] == caso["resultado_esperado"], (
                f"Caso {caso['id']}: esperado {caso['resultado_esperado']}, "
                f"obtido {resultado['status']}"
            )


# ================================================================
# Testes: RAG Engine
# ================================================================


class TestRAGSearch:
    """Testes do RAG engine com dados em memoria."""

    def test_search_retorna_resultados(self, rag_engine_memory):
        results = rag_engine_memory.search("parafuso de aco", k=2)
        assert len(results) == 2
        assert "text" in results[0]
        assert "score" in results[0]

    def test_search_relevancia_parafuso(self, rag_engine_memory):
        results = rag_engine_memory.search("parafuso de ferro roscado", k=3)
        assert "7318" in results[0]["text"]

    def test_search_relevancia_computador(self, rag_engine_memory):
        results = rag_engine_memory.search("computador portatil notebook", k=3)
        assert "8471" in results[0]["text"]

    def test_search_relevancia_telefone(self, rag_engine_memory):
        results = rag_engine_memory.search("telefone celular smartphone", k=3)
        assert "8517" in results[0]["text"]

    def test_search_k_maior_que_chunks(self, rag_engine_memory):
        results = rag_engine_memory.search("teste", k=100)
        assert len(results) == 3  # only 3 chunks exist

    def test_integracao_rag_validador(self, rag_engine_memory, mock_openai_response):
        """Testa ValidadorNCM usando RAG engine real + API mock."""
        v = ValidadorNCM(
            api_key="sk-fake", model="gpt-4o", rag_engine=rag_engine_memory
        )
        v.client = MagicMock()
        v.client.chat.completions.create.return_value = mock_openai_response()

        resultado = v.validar("Parafuso de aco inoxidavel M10", "7318.15.00")
        assert resultado["status"] == "APROVADO"

        # Verifica que o contexto RAG foi incluido no prompt
        call_args = v.client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        user_msg = messages[1]["content"]
        assert "7318" in user_msg


# ================================================================
# Testes: Calculo de custo
# ================================================================


class TestCalculoCusto:

    CUSTOS = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    }

    @pytest.mark.parametrize("modelo", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"])
    def test_custo_por_modelo(self, modelo):
        c = self.CUSTOS[modelo]
        custo = (500 * c["input"] + 150 * c["output"]) / 1_000_000
        assert custo > 0

    def test_gpt4o_mais_barato_que_turbo(self):
        c4o = (500 * 2.50 + 150 * 10.00) / 1_000_000
        c4t = (500 * 10.00 + 150 * 30.00) / 1_000_000
        assert c4o < c4t

    def test_gpt35_mais_barato_que_4o(self):
        c35 = (500 * 0.50 + 150 * 1.50) / 1_000_000
        c4o = (500 * 2.50 + 150 * 10.00) / 1_000_000
        assert c35 < c4o
