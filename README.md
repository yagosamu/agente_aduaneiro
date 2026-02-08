# Agente Aduaneiro - Auditor de Conformidade NCM

Agente de IA que audita a conformidade de classificacoes NCM (Nomenclatura Comum do Mercosul) utilizando GPT-4 como modelo base, RAG (Retrieval-Augmented Generation) sobre a NESH e interface interativa via Streamlit.

## O que o projeto faz

- Recebe descricoes de produtos e seus respectivos codigos NCM
- Consulta a base vetorizada da NESH (Notas Explicativas do Sistema Harmonizado) para contexto tecnico
- Utiliza GPT-4 para avaliar se a classificacao NCM esta correta
- Gera pareceres tecnicos com justificativa e sugestoes de correcao quando necessario

## Pre-requisitos

- Python 3.10+
- Chave de API da OpenAI

## Como obter a API Key da OpenAI

1. Acesse [platform.openai.com](https://platform.openai.com/)
2. Crie uma conta ou faca login
3. Va em **API Keys** no menu lateral
4. Clique em **Create new secret key**
5. Copie a chave gerada (ela so aparece uma vez)

## Instalacao

```bash
# Clone o repositorio
git clone <url-do-repositorio>
cd agente-aduaneiro

# Crie o ambiente virtual
python -m venv venv

# Ative o ambiente virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instale as dependencias
pip install -r requirements.txt
```

## Configuracao

```bash
# Copie o arquivo de exemplo e configure sua chave
cp .env.example .env
```

Edite o arquivo `.env` e insira sua chave da OpenAI:

```
OPENAI_API_KEY=sk-sua_chave_aqui
OPENAI_MODEL=gpt-4o
```

## Como rodar

```bash
streamlit run app.py
```

A aplicacao abrira no navegador em `http://localhost:8501`.

## Estrutura de pastas

```
agente-aduaneiro/
├── app.py                  # Aplicacao principal (Streamlit)
├── src/                    # Modulos do projeto
│   └── __init__.py
├── data/
│   ├── nesh/               # PDFs da NESH para indexacao
│   └── vectorstore/        # Base vetorial FAISS gerada
├── outputs/
│   └── logs/               # Logs de execucao e pareceres
├── requirements.txt        # Dependencias Python
├── .env.example            # Modelo de variaveis de ambiente
├── .gitignore              # Arquivos ignorados pelo Git
└── README.md               # Este arquivo
```
