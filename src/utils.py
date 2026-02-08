"""Funcoes auxiliares para o sistema de auditoria de conformidade aduaneira.

Modulo contendo utilitarios para validacao de NCM, calculo de custos,
contagem de tokens, logging de validacoes e exportacao de relatorios.
"""

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import tiktoken

# ================================================================
# Constantes
# ================================================================

LOG_DIR = Path("outputs/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

CUSTO_POR_1K_TOKENS: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
}

ENCODING_POR_MODELO: dict[str, str] = {
    "gpt-4o": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
}


# ================================================================
# 1. Validacao de formato NCM
# ================================================================


def validar_formato_ncm(ncm: str) -> bool:
    """Verifica se uma string NCM possui formato valido com 8 digitos.

    Aceita tanto o formato puro (8 digitos consecutivos) quanto o formato
    com separadores (pontos, tracos ou espacos). Remove separadores antes
    de verificar se restam exatamente 8 digitos numericos.

    Args:
        ncm: String contendo o codigo NCM a ser validado.

    Returns:
        True se o NCM contem exatamente 8 digitos numericos, False caso contrario.

    Examples:
        >>> validar_formato_ncm("73181500")
        True
        >>> validar_formato_ncm("7318.15.00")
        True
        >>> validar_formato_ncm("7318-15-00")
        True
        >>> validar_formato_ncm("7318.1.00")
        False
        >>> validar_formato_ncm("")
        False
    """
    digits = re.sub(r"[^0-9]", "", ncm)
    return len(digits) == 8


# ================================================================
# 2. Formatacao de NCM
# ================================================================


def formatar_ncm(ncm: str) -> str:
    """Formata um codigo NCM para o padrao XXXX.XX.XX.

    Remove todos os caracteres nao numericos e reformata os 8 digitos
    resultantes no padrao oficial com pontos separadores.

    Args:
        ncm: String contendo o codigo NCM em qualquer formato
             (ex: "73181500", "7318-15-00", "7318 15 00").

    Returns:
        String no formato "XXXX.XX.XX" se o NCM possuir 8 digitos,
        ou a string original sem espacos laterais caso contrario.

    Examples:
        >>> formatar_ncm("73181500")
        '7318.15.00'
        >>> formatar_ncm("7318-15-00")
        '7318.15.00'
        >>> formatar_ncm("7318 15 00")
        '7318.15.00'
        >>> formatar_ncm("09012100")
        '0901.21.00'
    """
    digits = re.sub(r"[^0-9]", "", ncm)
    if len(digits) == 8:
        return f"{digits[:4]}.{digits[4:6]}.{digits[6:8]}"
    return ncm.strip()


# ================================================================
# 3. Calculo de custo OpenAI
# ================================================================


def calcular_custo_openai(
    tokens_input: int, tokens_output: int, modelo: str
) -> float:
    """Calcula o custo em USD de uma chamada a API OpenAI.

    Utiliza a tabela de precos por 1K tokens para cada modelo suportado.
    Precos de referencia:
        - gpt-4o:        $0.005/1K input,  $0.015/1K output
        - gpt-4-turbo:   $0.01/1K input,   $0.03/1K output
        - gpt-3.5-turbo: $0.0005/1K input,  $0.0015/1K output

    Args:
        tokens_input: Numero de tokens de entrada (prompt).
        tokens_output: Numero de tokens de saida (completion).
        modelo: Nome do modelo OpenAI utilizado.

    Returns:
        Custo total em USD. Retorna 0.0 se o modelo nao for reconhecido.

    Examples:
        >>> calcular_custo_openai(1000, 500, "gpt-4o")
        0.0125
        >>> calcular_custo_openai(1000, 500, "gpt-3.5-turbo")
        0.00125
        >>> calcular_custo_openai(100, 50, "modelo-desconhecido")
        0.0
    """
    if modelo not in CUSTO_POR_1K_TOKENS:
        return 0.0
    custos = CUSTO_POR_1K_TOKENS[modelo]
    return (tokens_input * custos["input"] + tokens_output * custos["output"]) / 1_000


# ================================================================
# 4. Contagem de tokens
# ================================================================


def contar_tokens(texto: str, modelo: str = "gpt-4o") -> int:
    """Conta o numero de tokens em um texto usando tiktoken.

    Utiliza o encoding apropriado para o modelo especificado.
    Para modelos nao reconhecidos, usa cl100k_base como fallback.

    Args:
        texto: Texto a ter os tokens contados.
        modelo: Nome do modelo OpenAI para selecionar o encoding correto.
                Default: "gpt-4o".

    Returns:
        Numero inteiro de tokens no texto.

    Examples:
        >>> contar_tokens("Hello, world!")
        4
        >>> contar_tokens("Parafuso de aco inoxidavel M10", "gpt-4o")
        9
    """
    encoding_name = ENCODING_POR_MODELO.get(modelo, "cl100k_base")
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(texto))


# ================================================================
# 5. Salvar log de validacao
# ================================================================


def salvar_log_validacao(
    descricao: str,
    ncm: str,
    resultado: str,
    tokens_input: int,
    tokens_output: int,
    custo_usd: float,
    timestamp: str | None = None,
) -> Path:
    """Salva o registro de uma validacao em arquivo JSONL.

    Cada validacao e registrada como uma linha JSON no arquivo
    outputs/logs/validacoes_{data}.jsonl (um arquivo por dia).

    Args:
        descricao: Descricao do produto validado (truncada em 200 caracteres).
        ncm: Codigo NCM validado.
        resultado: Status da validacao (APROVADO, RISCO, ATENCAO, ERRO).
        tokens_input: Numero de tokens de entrada consumidos.
        tokens_output: Numero de tokens de saida consumidos.
        custo_usd: Custo da validacao em USD.
        timestamp: Timestamp ISO 8601. Se None, usa o horario atual.

    Returns:
        Path do arquivo de log onde o registro foi salvo.

    Examples:
        >>> salvar_log_validacao(
        ...     "Parafuso M10", "7318.15.00", "APROVADO",
        ...     500, 150, 0.00475
        ... )
        PosixPath('outputs/logs/validacoes_2026-02-08.jsonl')
    """
    if timestamp is None:
        timestamp = datetime.now().isoformat()

    ts_date = timestamp[:10]
    log_file = LOG_DIR / f"validacoes_{ts_date}.jsonl"

    entry = {
        "timestamp": timestamp,
        "descricao": descricao[:200],
        "ncm": ncm,
        "resultado": resultado,
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
        "custo_usd": custo_usd,
    }

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return log_file


# ================================================================
# 6. Carregar estatisticas
# ================================================================


def carregar_estatisticas() -> dict:
    """Le todos os arquivos de log e retorna estatisticas agregadas.

    Percorre todos os arquivos validacoes_*.jsonl em outputs/logs/
    e calcula totais de validacoes, tokens e custos.

    Returns:
        Dicionario com as seguintes chaves:
            - total (int): Numero total de validacoes registradas.
            - aprovados (int): Quantidade de validacoes APROVADO.
            - riscos (int): Quantidade de validacoes RISCO.
            - atencoes (int): Quantidade de validacoes ATENCAO.
            - erros (int): Quantidade de validacoes ERRO.
            - tokens_totais (int): Soma de todos os tokens (input + output).
            - custo_total (float): Soma de todos os custos em USD.

    Examples:
        >>> stats = carregar_estatisticas()
        >>> stats["total"]
        42
        >>> stats["custo_total"]
        0.1234
    """
    stats: dict = {
        "total": 0,
        "aprovados": 0,
        "riscos": 0,
        "atencoes": 0,
        "erros": 0,
        "tokens_totais": 0,
        "custo_total": 0.0,
    }

    contagem_map = {
        "APROVADO": "aprovados",
        "RISCO": "riscos",
        "ATENCAO": "atencoes",
        "ERRO": "erros",
    }

    for log_file in sorted(LOG_DIR.glob("validacoes_*.jsonl")):
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                stats["total"] += 1

                resultado = entry.get("resultado", entry.get("status", ""))
                chave = contagem_map.get(resultado)
                if chave:
                    stats[chave] += 1

                tokens_in = entry.get("tokens_input", entry.get("prompt_tokens", 0))
                tokens_out = entry.get(
                    "tokens_output", entry.get("completion_tokens", 0)
                )
                stats["tokens_totais"] += tokens_in + tokens_out
                stats["custo_total"] += entry.get("custo_usd", 0.0)

    return stats


# ================================================================
# 7. Exportar relatorio em lote
# ================================================================


def exportar_relatorio_lote(
    resultados: list[dict], formato: str = "csv"
) -> bytes:
    """Exporta resultados de validacao em lote para CSV ou Excel.

    Gera um arquivo binario (bytes) contendo a tabela de resultados
    com todas as colunas relevantes da validacao.

    Colunas incluidas: descricao, ncm, status, justificativa,
    ncm_alternativa, regras, tokens_input, tokens_output, custo_usd.

    Args:
        resultados: Lista de dicionarios, cada um representando o
                    resultado de uma validacao. Campos esperados:
                    descricao, ncm, status, mensagem, justificativa,
                    ncm_sugerida_alternativa, regras_citadas, usage.
        formato: Formato de saida: "csv" (default) ou "excel".

    Returns:
        Conteudo do arquivo como bytes, pronto para download ou gravacao.

    Raises:
        ValueError: Se o formato especificado nao for "csv" nem "excel".

    Examples:
        >>> dados = [{"descricao": "Parafuso", "ncm": "7318.15.00",
        ...           "status": "APROVADO", "mensagem": "OK",
        ...           "justificativa": "Correto", "regras_citadas": ["RGI 1"],
        ...           "ncm_sugerida_alternativa": None,
        ...           "usage": {"prompt_tokens": 500, "completion_tokens": 150}}]
        >>> csv_bytes = exportar_relatorio_lote(dados, "csv")
        >>> isinstance(csv_bytes, bytes)
        True
    """
    if formato not in ("csv", "excel"):
        raise ValueError(f"Formato invalido: '{formato}'. Use 'csv' ou 'excel'.")

    rows = []
    for r in resultados:
        usage = r.get("usage", {})
        t_in = usage.get("prompt_tokens", 0)
        t_out = usage.get("completion_tokens", 0)

        regras = r.get("regras_citadas", [])
        if isinstance(regras, list):
            regras_str = ", ".join(regras)
        else:
            regras_str = str(regras)

        rows.append(
            {
                "descricao": r.get("descricao", ""),
                "ncm": r.get("ncm", ""),
                "status": r.get("status", ""),
                "justificativa": r.get("justificativa", ""),
                "ncm_alternativa": r.get("ncm_sugerida_alternativa") or "",
                "regras": regras_str,
                "tokens_input": t_in,
                "tokens_output": t_out,
                "custo_usd": calcular_custo_openai(
                    t_in, t_out, r.get("modelo", "gpt-4o")
                ),
            }
        )

    df = pd.DataFrame(rows)

    if formato == "csv":
        return df.to_csv(index=False).encode("utf-8")
    else:
        import io

        buf = io.BytesIO()
        df.to_excel(buf, index=False, engine="openpyxl")
        return buf.getvalue()
