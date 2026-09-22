# Pipeline de Dados — Potencial de Microgeração Solar no Nordeste

Trabalho de Conclusão de Curso (TCC) que investiga, a nível **municipal**, onde
o Nordeste brasileiro tem maior *potencial inexplorado* de microgeração de
energia solar residencial — cruzando irradiação solar, renda per capita e
população com o número real de conexões solares já instaladas (ANEEL).

Pergunta de pesquisa: **dado o potencial solar, a renda e o tamanho da
população de um município, qual seria o número esperado de instalações de
microgeração residencial? E onde a adoção real está mais aquém desse
potencial?**

## Arquitetura

O projeto segue uma arquitetura de pipeline em camadas (medalhão:
Bronze → Prata → Ouro), implementada em notebooks Jupyter, com persistência
final em PostgreSQL:

```
Dados brutos (Excel/CSV)
        │
        ▼
   Extract.ipynb   ──► Camada Bronze  (Pipe/dados/bronze/*.pkl)
        │
        ▼
  Transform.ipynb   ──► Camada Prata → Camada Ouro
        │               (Pipe/dados/gold/dataset_energia_solar.parquet)
        ▼
    Load.ipynb      ──► PostgreSQL (tabela dataset_energia_solar)
        │
        ▼
    Model.py         ──► Modelagem estatística/ML, rankings e
                          Pipe/dados/dados.xlsx (resultado final)
```

- **Extract**: lê os dados brutos (renda per capita, ANEEL, IBGE, LABREN/INPE)
  e salva como pickle em `Pipe/dados/bronze/`, sem tratamento.
- **Transform**: limpa, normaliza nomes de municípios (removendo divergências
  de grafia/acentuação entre IBGE, ANEEL e LABREN), cruza as bases por
  `município + UF` e gera o dataset final municipal em
  `Pipe/dados/gold/dataset_energia_solar.parquet`. Inclui travas de
  integridade (`assert`) que interrompem o notebook se o merge falhar
  silenciosamente.
- **Load**: carrega a Camada Ouro no PostgreSQL (tabela
  `dataset_energia_solar`, banco `energia_solar`).
- **Model.py**: lê os dados do Postgres e roda a modelagem (regressão linear,
  Random Forest, Gradient Boosting), incluindo teste de ablação para detectar
  vazamento de dados, validação dos pressupostos da regressão (VIF,
  Breusch-Pagan, resíduos) e geração dos rankings de potencial inexplorado.

## Fontes de dados

| Fonte | Conteúdo | Nível |
|---|---|---|
| IBGE — Renda per capita (2022) | Renda per capita municipal | Município |
| IBGE — Censo/PNAD Contínua | População, área e densidade demográfica | Município |
| ANEEL | Empreendimentos de geração distribuída (conexões e potência instalada solar residencial) | Município |
| LABREN/INPE — Atlas Brasileiro de Energia Solar | Irradiação solar (GHI) média anual | Município (sede) |

Os arquivos brutos ficam em [`Dados/`](Dados/) (fontes originais, por ano/tema)
e [`Pipe/dados/`](Pipe/dados/) (cópias de trabalho usadas pelos notebooks).

## Estrutura do repositório

```
.
├── Dados/                     # Fontes de dados originais (IBGE, ANEEL, LABREN)
├── Pipe/
│   ├── notebooks/
│   │   ├── Extract.ipynb      # Camada Bronze
│   │   ├── Transform.ipynb    # Camadas Prata e Ouro
│   │   ├── Load.ipynb         # Carga no PostgreSQL
│   │   └── Model.py           # Modelagem e análise final
│   ├── dados/                 # Dados de trabalho (entrada + bronze/gold gerados)
│   └── Testes/                # Execuções de teste/validação da pipeline
├── Postgress/
│   ├── docker-compose.yml     # PostgreSQL + pgAdmin
│   └── postgres-init/         # Script de criação do banco
└── Pipeline Definições.docx   # Documento de definições do TCC
```

## Como executar

### 1. Subir o banco de dados

```bash
cd Postgress
docker compose up -d
```

Isso sobe:
- **PostgreSQL** em `localhost:5435` (banco `energia_solar`)
- **pgAdmin** em `localhost:5050`

> É necessário um arquivo `.env` em `Postgress/` com `POSTGRES_USER`,
> `POSTGRES_PASSWORD`, `POSTGRES_DB` (e, para o pgAdmin, `PGADMIN_DEFAULT_EMAIL`
> / `PGADMIN_DEFAULT_PASSWORD`), pois o `docker-compose.yml` os lê de lá.

### 2. Rodar a pipeline (Jupyter)

Os notebooks devem ser executados **em ordem**, a partir de `Pipe/notebooks/`:

1. `Extract.ipynb` — gera a Camada Bronze
2. `Transform.ipynb` — gera as Camadas Prata e Ouro
3. `Load.ipynb` — carrega os dados no PostgreSQL
4. `Model.py` — roda a modelagem e exporta os resultados finais

Cada notebook instala suas próprias dependências (`pip install ...` na
primeira célula): `pandas`, `openpyxl`, `pyarrow`, `fastparquet`,
`psycopg2-binary`, `sqlalchemy`, `numpy`, `scikit-learn`, `statsmodels`,
`matplotlib`, `seaborn`.

### 3. Resultados

Ao final da execução, `Model.py` gera em `Pipe/dados/`:
- `dados.xlsx` — dataset completo com previsões e potencial inexplorado
- `teste_ablacao.xlsx` — resultado do teste de ablação (comparação dos modelos)
- `municipios_prioritarios.xlsx` — ranking dos municípios prioritários para
  políticas públicas (alta irradiação + baixa renda + adoção aquém do esperado)

## Metodologia (resumo)

- **Variável alvo**: número de conexões solares residenciais por município
  (ANEEL).
- **Variáveis explicativas**: irradiação solar (GHI), renda per capita e
  população residente — todas estruturais e anteriores à decisão de instalar.
- **Potência instalada** é deliberadamente excluída do modelo por vazamento de
  dados (correlação ≈ 0,99 com o alvo); o teste de ablação em `Model.py`
  quantifica essa dependência.
- **Modelo final**: Random Forest, avaliado com validação cruzada K-Fold e
  previsões *out-of-fold*, para evitar sobreajuste e permitir comparar adoção
  real vs. esperada de forma honesta.
- **Potencial inexplorado** = conexões previstas − conexões reais, normalizado
  por mil habitantes para não ser dominado por capitais.

## Documentação relacionada

- [`Pipeline Definições.docx`](Pipeline%20Definições.docx) — definições e
  escopo do TCC.
- [`Dados/Microgeração Distribuida/Significado.txt`](Dados/Microgeração%20Distribuida/Significado.txt) —
  definição de microgeração distribuída (MMGD).
- [`Pipe/README.md`](Pipe/README.md) — acesso rápido ao Jupyter Lab.


# Pipe

Acesse o Jupyter Lab no seu navegador clicando ou acessando o link abaixo:
**[http://localhost:8888/](http://localhost:8888/)**

Os notebooks ficam em [`notebooks/`](notebooks/) e devem ser executados **nesta ordem**:

1. `Extract.ipynb` — lê os dados brutos e gera a Camada Bronze
2. `Transform.ipynb` — limpa, cruza as bases e gera a Camada Ouro
3. `Load.ipynb` — carrega a Camada Ouro no PostgreSQL (veja [`../Postgress/`](../Postgress/))
4. `Model.py` — modelagem, testes estatísticos e exportação dos resultados finais

Detalhes da pipeline, das fontes de dados e da metodologia estão no
[README principal do projeto](../README.md).
