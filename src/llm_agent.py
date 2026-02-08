import json
import logging
import time

from openai import OpenAI, APIError, RateLimitError, APIConnectionError

from src.prompts import SYSTEM_PROMPT_AUDITOR, USER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2


class ValidadorNCM:
    def __init__(self, api_key: str, model: str = "gpt-4o", rag_engine=None):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.rag_engine = rag_engine
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def validar(self, descricao: str, ncm: str) -> dict:
        """Valida a classificacao NCM de um produto.

        Args:
            descricao: Descricao do produto.
            ncm: Codigo NCM sugerido.

        Returns:
            Dict com status, mensagem, justificativa, ncm_sugerida_alternativa, regras_citadas.
        """
        # 1. Busca contexto via RAG
        regras_contexto = self._buscar_contexto_rag(descricao, ncm)

        # 2. Monta prompt
        user_message = USER_PROMPT_TEMPLATE.format(
            descricao_produto=descricao,
            ncm_sugerida=ncm,
            regras_contexto=regras_contexto,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_AUDITOR},
            {"role": "user", "content": user_message},
        ]

        logger.info("Enviando requisicao para %s...", self.model)

        # 3. Chama OpenAI
        response = self._call_openai(messages)

        # 4. Parseia resposta
        resultado = self._parse_response(response)

        resultado["usage"] = self.last_usage

        logger.info("Parecer: %s - %s", resultado.get("status"), resultado.get("mensagem"))
        return resultado

    def _buscar_contexto_rag(self, descricao: str, ncm: str) -> str:
        """Busca regras relevantes via RAG engine."""
        if self.rag_engine is None:
            return "Nenhuma base NESH disponivel para consulta."

        try:
            query = f"{descricao} NCM {ncm}"
            resultados = self.rag_engine.search(query, k=3)

            if not resultados:
                return "Nenhuma regra relevante encontrada na base NESH."

            contexto_parts = []
            for i, r in enumerate(resultados, 1):
                contexto_parts.append(f"[Trecho {i} (score: {r['score']:.4f})]:\n{r['text']}")

            return "\n\n".join(contexto_parts)

        except Exception:
            logger.exception("Erro ao buscar contexto RAG")
            return "Erro ao consultar base NESH. Analise sem contexto adicional."

    def _call_openai(self, messages: list[dict]) -> str:
        """Chama a API da OpenAI com retry para rate limits."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                self.last_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                logger.info(
                    "Resposta recebida (tokens: prompt=%d, completion=%d)",
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )
                return content

            except RateLimitError as e:
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        "Rate limit atingido. Tentativa %d/%d. Aguardando %ds...",
                        attempt, MAX_RETRIES, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error("Rate limit: tentativas esgotadas.")
                    raise

            except APIConnectionError as e:
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * attempt
                    logger.warning(
                        "Erro de conexao. Tentativa %d/%d. Aguardando %ds...",
                        attempt, MAX_RETRIES, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error("Erro de conexao: tentativas esgotadas.")
                    raise

            except APIError as e:
                logger.error("Erro na API OpenAI: %s", e)
                raise

    def _parse_response(self, response: str) -> dict:
        """Extrai e valida o JSON da resposta."""
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            logger.error("Resposta nao e JSON valido: %s", response[:200])
            return {
                "status": "ERRO",
                "mensagem": "Falha ao interpretar resposta do modelo.",
                "justificativa": f"Resposta bruta: {response[:500]}",
                "ncm_sugerida_alternativa": None,
                "regras_citadas": [],
            }

        campos_obrigatorios = ["status", "mensagem", "justificativa", "regras_citadas"]
        for campo in campos_obrigatorios:
            if campo not in data:
                data[campo] = "" if campo != "regras_citadas" else []

        if "ncm_sugerida_alternativa" not in data:
            data["ncm_sugerida_alternativa"] = None

        return data
