import streamlit as st
import os
import re
import json
import io
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import AuthenticationError

from src.rag_engine import RAGEngine
from src.llm_agent import ValidadorNCM

load_dotenv()

# ================================================================
# Constants
# ================================================================

MODELOS = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]

CUSTO_POR_1M_TOKENS = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
}

LOG_DIR = Path("outputs/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================
# Page Config
# ================================================================

st.set_page_config(
    page_title="Auditor de Conformidade Aduaneiro",
    page_icon="\U0001f6c3",
    layout="wide",
)

# ================================================================
# Session State
# ================================================================

_defaults = {
    "historico": [],
    "resultado_lote": None,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ================================================================
# Cached Resources
# ================================================================


@st.cache_resource
def carregar_rag_engine():
    engine = RAGEngine()
    try:
        engine.load_vectorstore()
        return engine, True
    except FileNotFoundError:
        return engine, False


@st.cache_resource
def criar_validador(api_key, modelo, usar_rag):
    rag_engine, rag_ok = carregar_rag_engine()
    engine = rag_engine if (usar_rag and rag_ok) else None
    return ValidadorNCM(api_key=api_key, model=modelo, rag_engine=engine)


# ================================================================
# Helpers
# ================================================================


def normalizar_ncm(ncm_raw: str) -> str:
    digits = re.sub(r"[^0-9]", "", ncm_raw)
    if len(digits) == 8:
        return f"{digits[:4]}.{digits[4:6]}.{digits[6:8]}"
    return ncm_raw.strip()


def validar_formato_ncm(ncm: str) -> bool:
    return bool(re.match(r"^\d{4}\.\d{2}\.\d{2}$", ncm))


ALIASES_DESCRICAO = {
    "descricao", "descrição", "description", "desc", "produto",
    "product", "mercadoria", "item", "nome", "name",
}

ALIASES_NCM = {
    "ncm", "codigo_ncm", "código_ncm", "ncm_code", "cod_ncm",
    "código ncm", "codigo ncm", "cod ncm",
}


def _remover_acentos(texto: str) -> str:
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _normalizar_nome_coluna(nome: str) -> str:
    return _remover_acentos(nome.strip().lower())


def detectar_colunas(df: pd.DataFrame, uploaded_file) -> pd.DataFrame | None:
    """Detecta e renomeia colunas 'descricao' e 'ncm' usando aliases conhecidos.

    Tenta primeiro mapear aliases nos nomes de colunas existentes.
    Se falhar, escaneia as primeiras 10 linhas procurando uma linha que
    sirva como cabecalho e re-le o arquivo usando skiprows.

    Retorna o DataFrame com colunas renomeadas ou None se nao conseguir detectar.
    """
    col_descricao = None
    col_ncm = None

    # Fase 1: mapear aliases nos nomes de colunas atuais
    for col_original in df.columns:
        col_norm = _normalizar_nome_coluna(str(col_original))
        if col_norm in ALIASES_DESCRICAO and col_descricao is None:
            col_descricao = col_original
        elif col_norm in ALIASES_NCM and col_ncm is None:
            col_ncm = col_original

    if col_descricao and col_ncm:
        df = df.rename(columns={col_descricao: "descricao", col_ncm: "ncm"})
        df = df.loc[:, ~df.columns.duplicated()]
        return df

    # Fase 2: fallback — procurar cabecalho nas primeiras 50 linhas
    aliases_todos = ALIASES_DESCRICAO | ALIASES_NCM
    max_linhas = min(50, len(df))

    for idx in range(max_linhas):
        valores = [_normalizar_nome_coluna(str(v)) for v in df.iloc[idx]]
        matches = sum(1 for v in valores if v in aliases_todos)
        if matches >= 2:
            # Encontrou uma linha que parece ser cabecalho
            uploaded_file.seek(0)
            skiprows = idx + 1  # +1 porque pandas ja pulou a primeira linha como header
            if uploaded_file.name.endswith(".csv"):
                df_novo = pd.read_csv(uploaded_file, skiprows=skiprows, header=0)
            else:
                df_novo = pd.read_excel(uploaded_file, skiprows=skiprows, header=0)

            # Renomear o novo header usando a linha encontrada
            novos_nomes = [str(v).strip() for v in df.iloc[idx]]
            if len(novos_nomes) == len(df_novo.columns):
                df_novo.columns = novos_nomes

            # Tentar mapear aliases novamente
            col_descricao = None
            col_ncm = None
            for col_original in df_novo.columns:
                col_norm = _normalizar_nome_coluna(str(col_original))
                if col_norm in ALIASES_DESCRICAO and col_descricao is None:
                    col_descricao = col_original
                elif col_norm in ALIASES_NCM and col_ncm is None:
                    col_ncm = col_original

            if col_descricao and col_ncm:
                df_novo = df_novo.rename(
                    columns={col_descricao: "descricao", col_ncm: "ncm"}
                )
                df_novo = df_novo.loc[:, ~df_novo.columns.duplicated()]
                return df_novo

    return None


def calcular_custo(prompt_tokens: int, completion_tokens: int, modelo: str) -> float:
    if modelo not in CUSTO_POR_1M_TOKENS:
        return 0.0
    c = CUSTO_POR_1M_TOKENS[modelo]
    return (prompt_tokens * c["input"] + completion_tokens * c["output"]) / 1_000_000


def registrar_log(resultado: dict, modelo: str, descricao: str, ncm: str):
    log_file = LOG_DIR / f"validacoes_{datetime.now().strftime('%Y-%m-%d')}.jsonl"
    usage = resultado.get("usage", {})
    entry = {
        "timestamp": datetime.now().isoformat(),
        "modelo": modelo,
        "descricao": descricao[:200],
        "ncm": ncm,
        "status": resultado.get("status"),
        "mensagem": resultado.get("mensagem", ""),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "custo_usd": calcular_custo(
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            modelo,
        ),
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def carregar_historico_logs() -> pd.DataFrame:
    registros = []
    for log_file in sorted(LOG_DIR.glob("validacoes_*.jsonl")):
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        registros.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    if not registros:
        return pd.DataFrame()
    return pd.DataFrame(registros)


def exibir_resultado(resultado: dict, modelo: str):
    status = resultado.get("status", "ERRO")
    mensagem = resultado.get("mensagem", "")
    justificativa = resultado.get("justificativa", "")
    ncm_alt = resultado.get("ncm_sugerida_alternativa")
    regras = resultado.get("regras_citadas", [])
    usage = resultado.get("usage", {})

    if status == "APROVADO":
        st.success(f"**APROVADO** — {mensagem}")
    elif status == "ATENCAO":
        st.warning(f"**ATENCAO** — {mensagem}")
    elif status == "RISCO":
        st.error(f"**RISCO** — {mensagem}")
    else:
        st.info(f"**{status}** — {mensagem}")

    st.markdown(f"**Justificativa:** {justificativa}")

    if ncm_alt and str(ncm_alt).lower() not in ("null", "none", ""):
        st.info(f"**NCM alternativa sugerida:** `{ncm_alt}`")

    if regras:
        with st.expander("Regras Citadas"):
            for regra in regras:
                st.markdown(f"- {regra}")

    prompt_t = usage.get("prompt_tokens", 0)
    compl_t = usage.get("completion_tokens", 0)
    custo = calcular_custo(prompt_t, compl_t, modelo)

    c1, c2, c3 = st.columns(3)
    c1.metric("Tokens (prompt)", f"{prompt_t:,}")
    c2.metric("Tokens (completion)", f"{compl_t:,}")
    c3.metric("Custo estimado", f"${custo:.6f}")


# ================================================================
# Sidebar
# ================================================================

with st.sidebar:
    st.header("Configuracoes")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Chave da API OpenAI (sk-...)",
    )

    modelo = st.selectbox("Modelo OpenAI", MODELOS, index=0)

    if modelo in CUSTO_POR_1M_TOKENS:
        c = CUSTO_POR_1M_TOKENS[modelo]
        st.caption(
            f"Custo: ${c['input']:.2f}/1M input | ${c['output']:.2f}/1M output"
        )

    st.divider()

    st.subheader("Base NESH")

    usar_rag = st.checkbox("Usar vector store existente", value=True)

    if st.button("Reprocessar documentos NESH", use_container_width=True):
        with st.spinner("Reprocessando base NESH..."):
            try:
                engine_tmp = RAGEngine()
                engine_tmp.build_vectorstore()
                st.success(f"Base reprocessada: {len(engine_tmp.chunks)} chunks.")
                carregar_rag_engine.clear()
                criar_validador.clear()
                st.rerun()
            except FileNotFoundError as e:
                st.error(f"Erro: {e}")
            except Exception as e:
                st.error(f"Erro ao reprocessar: {e}")

    if usar_rag:
        _rag, _rag_ok = carregar_rag_engine()
        if _rag_ok:
            st.success(f"Base NESH carregada: {_rag.index.ntotal} vetores")
        else:
            st.warning("Base NESH nao encontrada. Coloque PDFs em data/nesh/ e reprocesse.")
    else:
        st.info("Validacao sem base NESH.")

    st.divider()

    st.subheader("Metricas da Sessao")
    st.metric("Validacoes realizadas", len(st.session_state.historico))

    custo_sessao = sum(
        calcular_custo(
            h.get("usage", {}).get("prompt_tokens", 0),
            h.get("usage", {}).get("completion_tokens", 0),
            modelo,
        )
        for h in st.session_state.historico
    )
    st.metric("Custo estimado (sessao)", f"${custo_sessao:.4f}")


# ================================================================
# Header
# ================================================================

st.title("\U0001f6c3 Auditor de Conformidade Aduaneiro")
st.caption("Powered by GPT-4 | Validacao de classificacao NCM com base NESH")

# ================================================================
# Tabs
# ================================================================

tab1, tab2, tab3 = st.tabs(
    ["Validacao Unica", "Validacao em Lote", "Custos"]
)

# ----------------------------------------------------------------
# TAB 1 — Validacao Unica
# ----------------------------------------------------------------

with tab1:
    col_desc, col_ncm = st.columns([3, 1])

    with col_desc:
        descricao = st.text_area(
            "Descricao do Produto *",
            height=120,
            placeholder="Ex: Parafuso de aco inoxidavel, cabeca sextavada, M10x50mm, para uso industrial",
        )

    with col_ncm:
        ncm_raw = st.text_input(
            "Codigo NCM *",
            placeholder="7318.15.00",
            help="8 digitos: XXXX.XX.XX",
        )
        if ncm_raw:
            ncm_fmt = normalizar_ncm(ncm_raw)
            if validar_formato_ncm(ncm_fmt):
                st.caption(f"NCM: `{ncm_fmt}`")
            else:
                st.caption("Formato invalido. Use 8 digitos.")

    if st.button("Validar Conformidade", type="primary", use_container_width=True):
        if not api_key or not api_key.startswith("sk-"):
            st.error("Configure uma API Key valida na barra lateral.")
        elif not descricao.strip():
            st.warning("Preencha a descricao do produto.")
        elif not ncm_raw.strip():
            st.warning("Preencha o codigo NCM.")
        else:
            ncm = normalizar_ncm(ncm_raw)
            if not validar_formato_ncm(ncm):
                st.error(
                    f"Formato NCM invalido: `{ncm_raw}`. Use 8 digitos (ex: 7318.15.00)."
                )
            else:
                with st.spinner(f"Analisando via {modelo}..."):
                    try:
                        validador = criar_validador(api_key, modelo, usar_rag)
                        resultado = validador.validar(descricao.strip(), ncm)

                        registrar_log(resultado, modelo, descricao.strip(), ncm)
                        st.session_state.historico.append(resultado)

                        st.markdown("---")
                        st.subheader("Parecer Tecnico")
                        exibir_resultado(resultado, modelo)

                    except AuthenticationError:
                        st.error("API Key invalida. Verifique na barra lateral.")
                    except Exception as e:
                        msg = str(e).lower()
                        if "rate_limit" in msg or "rate limit" in msg:
                            st.error(
                                "Rate limit atingido. Aguarde alguns segundos e tente novamente."
                            )
                        else:
                            st.error(f"Erro na validacao: {e}")

# ----------------------------------------------------------------
# TAB 2 — Validacao em Lote
# ----------------------------------------------------------------

with tab2:
    st.markdown(
        "Envie um arquivo **CSV** ou **Excel** com colunas de descricao do produto e codigo NCM. "
        "Aceitamos variacoes como `descricao`, `produto`, `description`, `item` "
        "e `ncm`, `ncm_code`, `codigo_ncm`, entre outros."
    )

    uploaded_file = st.file_uploader(
        "Selecione o arquivo",
        type=["csv", "xlsx", "xls"],
    )

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_input = pd.read_csv(uploaded_file)
            else:
                df_input = pd.read_excel(uploaded_file)

            df_input = detectar_colunas(df_input, uploaded_file)

            if df_input is not None:
                df_input = df_input.dropna(subset=["descricao", "ncm"], how="all")
                df_input = df_input[
                    df_input["descricao"].astype(str).str.strip().ne("")
                    & df_input["ncm"].astype(str).str.strip().ne("")
                    & df_input["descricao"].astype(str).str.lower().ne("nan")
                    & df_input["ncm"].astype(str).str.lower().ne("nan")
                ].reset_index(drop=True)

            if df_input is None:
                st.error(
                    "Nao foi possivel detectar as colunas de descricao e NCM. "
                    "Use nomes como 'descricao', 'produto', 'description' e "
                    "'ncm', 'ncm_code', 'codigo_ncm'."
                )
            else:
                st.dataframe(
                    df_input[["descricao", "ncm"]].head(10), use_container_width=True
                )
                st.caption(f"Total de registros: {len(df_input)}")

                if st.button(
                    "Processar Lote", type="primary", use_container_width=True
                ):
                    if not api_key or not api_key.startswith("sk-"):
                        st.error("Configure uma API Key valida na barra lateral.")
                    else:
                        validador = criar_validador(api_key, modelo, usar_rag)

                        resultados_lote = []
                        total_prompt = 0
                        total_completion = 0

                        progress = st.progress(0, text="Processando...")
                        status_text = st.empty()

                        for i, row in df_input.iterrows():
                            desc = str(row["descricao"]).strip()
                            ncm_b = normalizar_ncm(str(row["ncm"]).strip())

                            status_text.text(
                                f"Processando {i + 1}/{len(df_input)}: {desc[:50]}..."
                            )

                            try:
                                if not validar_formato_ncm(ncm_b):
                                    res = {
                                        "status": "ERRO",
                                        "mensagem": f"NCM invalido: {row['ncm']}",
                                        "justificativa": "",
                                        "ncm_sugerida_alternativa": None,
                                        "regras_citadas": [],
                                        "usage": {},
                                    }
                                else:
                                    res = validador.validar(desc, ncm_b)
                                    usage = res.get("usage", {})
                                    total_prompt += usage.get("prompt_tokens", 0)
                                    total_completion += usage.get(
                                        "completion_tokens", 0
                                    )
                                    registrar_log(res, modelo, desc, ncm_b)
                                    st.session_state.historico.append(res)
                            except Exception as e:
                                res = {
                                    "status": "ERRO",
                                    "mensagem": str(e)[:200],
                                    "justificativa": "",
                                    "ncm_sugerida_alternativa": None,
                                    "regras_citadas": [],
                                    "usage": {},
                                }

                            resultados_lote.append(
                                {
                                    "descricao": desc,
                                    "ncm": ncm_b,
                                    "status": res.get("status"),
                                    "mensagem": res.get("mensagem"),
                                    "justificativa": res.get("justificativa", ""),
                                    "ncm_alternativa": res.get(
                                        "ncm_sugerida_alternativa"
                                    )
                                    or "",
                                    "regras": ", ".join(
                                        res.get("regras_citadas", [])
                                    ),
                                }
                            )

                            progress.progress((i + 1) / len(df_input))

                        status_text.empty()
                        progress.empty()

                        st.success(
                            f"Lote processado: {len(resultados_lote)} validacoes."
                        )

                        df_res = pd.DataFrame(resultados_lote)
                        st.session_state.resultado_lote = df_res

                        custo_lote = calcular_custo(
                            total_prompt, total_completion, modelo
                        )
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Tokens (prompt)", f"{total_prompt:,}")
                        c2.metric("Tokens (completion)", f"{total_completion:,}")
                        c3.metric("Custo total estimado", f"${custo_lote:.4f}")

        except Exception as e:
            st.error(f"Erro ao ler arquivo: {e}")

    if st.session_state.resultado_lote is not None:
        df_res = st.session_state.resultado_lote

        def _cor_status(val):
            cores = {
                "APROVADO": "background-color: #d4edda",
                "ATENCAO": "background-color: #fff3cd",
                "RISCO": "background-color: #f8d7da",
                "ERRO": "background-color: #e2e3e5",
            }
            return cores.get(val, "")

        styled = df_res.style.map(_cor_status, subset=["status"])
        st.dataframe(styled, use_container_width=True)

        csv_buf = io.StringIO()
        df_res.to_csv(csv_buf, index=False)
        st.download_button(
            "Download Relatorio (CSV)",
            data=csv_buf.getvalue(),
            file_name=f"relatorio_ncm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ----------------------------------------------------------------
# TAB 3 — Custos
# ----------------------------------------------------------------

with tab3:
    st.subheader("Tabela de Precos por Modelo")

    df_precos = pd.DataFrame(
        [
            {
                "Modelo": m,
                "Input (USD/1M tokens)": f"${c['input']:.2f}",
                "Output (USD/1M tokens)": f"${c['output']:.2f}",
            }
            for m, c in CUSTO_POR_1M_TOKENS.items()
        ]
    )
    st.table(df_precos)

    st.divider()
    st.subheader("Historico de Custos")

    df_hist = carregar_historico_logs()

    if df_hist.empty:
        st.info("Nenhuma validacao registrada ainda.")
    else:
        df_hist["data"] = pd.to_datetime(df_hist["timestamp"], format="ISO8601").dt.date

        df_diario = (
            df_hist.groupby("data")
            .agg(
                validacoes=("status", "count"),
                prompt_tokens=("prompt_tokens", "sum"),
                completion_tokens=("completion_tokens", "sum"),
                custo_usd=("custo_usd", "sum"),
            )
            .reset_index()
        )

        st.dataframe(df_diario, use_container_width=True)

        st.subheader("Tokens utilizados por dia")
        chart_data = df_diario[
            ["data", "prompt_tokens", "completion_tokens"]
        ].set_index("data")
        st.bar_chart(chart_data)

        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total validacoes", f"{len(df_hist):,}")
        total_tk = df_hist["prompt_tokens"].sum() + df_hist["completion_tokens"].sum()
        c2.metric("Total tokens", f"{total_tk:,.0f}")
        c3.metric("Custo total", f"${df_hist['custo_usd'].sum():.4f}")
        c4.metric("Custo medio", f"${df_hist['custo_usd'].mean():.6f}")
