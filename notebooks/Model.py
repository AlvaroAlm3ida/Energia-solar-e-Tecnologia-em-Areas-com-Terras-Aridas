# Bibliotecas
# pip install pandas numpy matplotlib seaborn scikit-learn statsmodels sqlalchemy psycopg2-binary
import os
import numpy as np                                                             # operações numéricas (regressão da linha de tendência, log, etc.)
import pandas as pd                                                            # manipulação do dataset em formato de tabela
import matplotlib.pyplot as plt                                                # geração dos gráficos
import seaborn as sns                                                          # heatmap de correlação
from sqlalchemy import create_engine                                           # conexão com o Postgres (Camada Ouro)
from sklearn.linear_model import LinearRegression                              # modelo 1: regressão linear
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor  # modelos 2 e 3: ensembles de árvores
from sklearn.model_selection import (train_test_split, cross_val_score,        # separação treino/teste e validação cruzada
                                     cross_val_predict, KFold)
from sklearn.preprocessing import StandardScaler                               # padronização (coeficientes comparáveis)
from sklearn.pipeline import make_pipeline                                     # encadeia padronização + modelo sem vazamento
from sklearn.metrics import mean_squared_error, r2_score                       # métricas de avaliação (RMSE e R²)
import statsmodels.api as sm                                                   # sumário estatístico com p-valores
from statsmodels.stats.diagnostic import het_breuschpagan                      # teste de homocedasticidade
from statsmodels.stats.outliers_influence import variance_inflation_factor     # teste de multicolinearidade (VIF)

# Paleta usada nos gráficos deste notebook (mantida fixa para todos eles)
COR_TINTA       = "#0b0b0b"   # texto/títulos
COR_TINTA_FRACA = "#898781"   # eixos e rótulos secundários
COR_GRADE       = "#e1e0d9"   # linhas de grade e bordas
COR_PONTO       = "#2a78d6"   # cor principal dos pontos (municípios)
COR_TENDENCIA   = "#e34948"   # linha de tendência / referência


def estilizar(ax):
    """Aplica o padrao visual dos graficos a um eixo, evitando repetir o bloco."""
    ax.tick_params(colors=COR_TINTA_FRACA, labelsize=9)
    ax.grid(True, color=COR_GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ["top", "right"]:
        ax.spines[lado].set_visible(False)
    for lado in ["left", "bottom"]:
        ax.spines[lado].set_color(COR_GRADE)


# %%
# PASSO 1 — Carregar os dados
# Le o dataset final (nivel MUNICIPAL) gerado pelo Load.ipynb, ja com todas as
# variaveis prontas (renda, populacao, densidade, GHI municipal, conexoes etc.)
engine = create_engine(
    "postgresql://admin:admin123@localhost:5435/energia_solar"
)

df = pd.read_sql("SELECT * FROM dataset_energia_solar", engine)
print(f"Dataset carregado: {len(df)} municípios")

# Fernando de Noronha (PE) fica sem GHI: o grid do LABREN cobre apenas o
# territorio continental e o arquipelago esta a ~350 km da costa. Como e 1
# municipio em 1.794, e removido da modelagem com a justificativa documentada.
antes = len(df)
df = df.dropna(subset=["ghi_media_kwh_m2_dia"]).reset_index(drop=True)
if len(df) < antes:
    print(f"Removido(s) {antes - len(df)} município(s) sem GHI municipal")

# Checagem de integridade das demais variaveis.
#
# ATENCAO - por que isto interrompe a execucao em vez de apenas descartar:
# descartar linhas com NaN silenciosamente esconde falha de merge. Numa
# execucao anterior, uma divergencia na regra de normalizacao de nomes entre
# as camadas fez 769 municipios (43% da base) perderem populacao. O modelo
# rodou normalmente com 1.024 municipios e produziu R² de 0,50 com desvio de
# +/-0,22 - numeros plausiveis o bastante para nao levantar suspeita, mas
# calculados sobre uma amostra mutilada e enviesada (so os municipios sem
# acento no nome sobreviveram).
#
# Perda de ate 1% e tolerada como ruido de cadastro. Acima disso, o problema
# esta na pipeline e precisa ser corrigido no Transform, nao contornado aqui.
COLUNAS_MODELO = [
    "ghi_media_kwh_m2_dia", "renda_per_capita", "populacao_residente",
    "densidade_demografica", "Qtd_Conexoes_Total", "Potencia_Instalada_KW_Total",
]
faltantes = df[COLUNAS_MODELO].isna().sum()
faltantes = faltantes[faltantes > 0]

if not faltantes.empty:
    perdidos = df[COLUNAS_MODELO].isna().any(axis=1).sum()
    proporcao = perdidos / len(df)

    print("\nValores ausentes por coluna:")
    print(faltantes.to_string())
    print(f"Municípios afetados: {perdidos} ({proporcao:.1%})")

    if proporcao > 0.01:
        raise ValueError(
            f"\n{'='*60}\n"
            f"PIPELINE INCOMPLETA — {perdidos} municípios ({proporcao:.1%}) sem dados.\n"
            f"{'='*60}\n"
            f"Colunas afetadas:\n{faltantes.to_string()}\n\n"
            f"Perda desse tamanho indica merge malsucedido no Transform.ipynb,\n"
            f"não ausência real de dado. Verifique se todas as chaves de merge\n"
            f"usam normalizar() e rode Transform e Load novamente antes de\n"
            f"prosseguir. Resultados obtidos sobre a amostra reduzida NÃO são\n"
            f"válidos para o TCC."
        )
    print("Perda abaixo de 1% — tratada como ruído de cadastro.")

antes = len(df)
df = df.dropna(subset=COLUNAS_MODELO).reset_index(drop=True)
if len(df) < antes:
    print(f"Removido(s) {antes - len(df)} município(s) com dados ausentes")

# A base do Nordeste tem 1.794 municípios; 1.793 entram no modelo (Fernando de
# Noronha sai por falta de GHI). Divergir disso é sinal de problema a montante.
print(f"Municípios na modelagem: {len(df)}")
if len(df) < 1780:
    print(f"AVISO: esperados ~1.793 municípios. Confira a execução do Transform.")


# %%
# PASSO 2 — Definir a pergunta de pesquisa e as variáveis
# Pergunta: dado o potencial solar, a renda e o tamanho da populacao de um
# municipio, qual seria o numero esperado de instalacoes de microgeracao?
#
# Variavel alvo (Y):
#   Qtd_Conexoes_Total ....... nº de conexoes solares residenciais (ANEEL)
#
# Variaveis de entrada (X) — sao fatores ESTRUTURAIS, anteriores e independentes
# da decisao de instalar. E isso que permite ler a previsao como "potencial":
#   ghi_media_kwh_m2_dia ..... potencial solar da sede municipal (LABREN/INPE)
#   renda_per_capita ......... poder de compra para investir em painel (IBGE)
#   populacao_residente ...... tamanho do mercado potencial (IBGE)
#
# VARIAVEL DESCARTADA — Potencia_Instalada_KW_Total:
#   Vem da MESMA tabela da ANEEL que a variavel alvo. Qtd_Conexoes_Total e o
#   count() das instalacoes; Potencia_Instalada_KW_Total e o sum() da potencia
#   dessas mesmas instalacoes. Sao duas medidas do mesmo fenomeno (adocao solar
#   ja existente), ligadas por uma relacao quase aritmetica.
#   Usa-la como preditora configura VAZAMENTO DE DADOS (data leakage): o modelo
#   aprende a converter kW em nº de conexoes, em vez de aprender a relacao real
#   com sol, renda e populacao. O R² sobe, mas o modelo perde utilidade — ele
#   passa a exigir, como entrada, um dado que so existe DEPOIS da adocao que
#   deveria prever. O PASSO 4 comprova isso numericamente.

VAR_ALVO      = "Qtd_Conexoes_Total"
VAR_VAZAMENTO = "Potencia_Instalada_KW_Total"

VARIAVEIS_FINAIS = [
    "ghi_media_kwh_m2_dia",
    "renda_per_capita",
    "populacao_residente",
]

ROTULOS = {
    "ghi_media_kwh_m2_dia":        "GHI Médio (kWh/m²/dia)",
    "renda_per_capita":            "Renda per Capita (R$)",
    "populacao_residente":         "População Residente (hab)",
    "densidade_demografica":       "Densidade Demográfica (hab/km²)",
    "Potencia_Instalada_KW_Total": "Potência Instalada (kW)",
}

X = df[VARIAVEIS_FINAIS]
y = df[VAR_ALVO]

print(f"\nVariáveis de entrada (X): {VARIAVEIS_FINAIS}")
print(f"Variável alvo (Y): {VAR_ALVO}")


# %%
# PASSO 3 — Explorar a correlação entre as variáveis
# Serve para checar, antes de treinar, se as variaveis de entrada realmente se
# relacionam com o alvo — e, principalmente, para expor a correlacao quase
# perfeita entre a potencia instalada e o alvo, que motiva o teste do PASSO 4.
colunas_corr = VARIAVEIS_FINAIS + ["densidade_demografica", VAR_VAZAMENTO, VAR_ALVO]

plt.figure(figsize=(9, 7))
sns.heatmap(
    df[colunas_corr].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0, vmin=-1, vmax=1
)
plt.title("Correlação entre variáveis")
plt.tight_layout()
plt.show()

r_vazamento = df[VAR_VAZAMENTO].corr(df[VAR_ALVO])
print("=" * 60)
print("DIAGNÓSTICO DE VAZAMENTO DE DADOS")
print("=" * 60)
print(f"Correlação entre {VAR_VAZAMENTO} e {VAR_ALVO}: r = {r_vazamento:.4f}")
if abs(r_vazamento) > 0.90:
    print("\n>> r > 0,90 confirma que as duas variáveis medem o mesmo fenômeno.")
    print(">> A variável é candidata a remoção — testada formalmente no PASSO 4.")


# %%
# PASSO 4 — TESTE DE ABLAÇÃO
# Ablacao = remover uma variavel de propósito e medir o quanto o modelo piora.
# E o teste que transforma a decisao de descartar a potencia instalada de uma
# escolha subjetiva em uma decisao metodologica com evidencia numerica.
#
# Se o R² despencar da configuracao A para a C, fica demonstrado que o
# desempenho alto vinha do dado circular, e nao de poder explicativo real.

CONFIGURACOES = {
    "A. Com vazamento (todas)": [
        "ghi_media_kwh_m2_dia", "renda_per_capita",
        "populacao_residente", "densidade_demografica", VAR_VAZAMENTO,
    ],
    "B. Sem vazamento (+ densidade)": [
        "ghi_media_kwh_m2_dia", "renda_per_capita",
        "populacao_residente", "densidade_demografica",
    ],
    "C. Modelo final (GHI+Renda+Pop)": VARIAVEIS_FINAIS,
}

# Testamos 3 abordagens diferentes para ver qual generaliza melhor:
# uma linear (simples, interpretável) e duas em ensemble de árvores
# (capturam relações não-lineares, ao custo de serem "caixa-preta")
modelos = {
    "Regressão Linear Múltipla": LinearRegression(),
    "Random Forest Regressor":   RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting":         GradientBoostingRegressor(random_state=42)
}

# K-Fold: em vez de uma unica divisao 80/20 (que depende do random_state
# sorteado), divide os dados em 5 partes e treina 5 vezes, cada uma usando uma
# parte diferente como teste. O R² medio das 5 rodadas e uma estimativa bem
# mais estavel — e o desvio mostra se o modelo e consistente.
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

resultados = []

for nome_config, variaveis in CONFIGURACOES.items():
    X_cfg = df[variaveis]

    # 80% treino / 20% teste; random_state fixo para o resultado ser reprodutível
    X_train, X_test, y_train, y_test = train_test_split(
        X_cfg, y, test_size=0.2, random_state=42
    )

    for nome_modelo, modelo in modelos.items():
        modelo.fit(X_train, y_train)      # treina só com os dados de treino
        pred = modelo.predict(X_test)     # prevê no teste (dados que nunca viu)

        scores_cv = cross_val_score(modelo, X_cfg, y, cv=kfold, scoring="r2")

        # RMSE calculado sobre as previsoes OUT-OF-FOLD, e nao sobre o holdout.
        #
        # Motivo: a divisao 80/20 com random_state=42 nao sorteou NENHUM dos 14
        # municipios com mais de 10 mil conexoes para o conjunto de teste. O
        # RMSE do holdout ficava em ~309 conexoes simplesmente porque o modelo
        # nunca precisou prever um caso dificil - numero otimista e enganoso.
        # No out-of-fold todo municipio e previsto por um modelo que nao o viu,
        # inclusive as capitais, e o RMSE sobe para ~870. Este e o honesto.
        pred_oof = cross_val_predict(modelo, X_cfg, y, cv=kfold)

        # RMSE: erro médio em "quantidade de conexões" (mesma unidade do alvo, menor é melhor)
        # R²: fração da variação do alvo que o modelo explica (0 a 1, maior é melhor)
        resultados.append({
            "Configuração":     nome_config,
            "Modelo":           nome_modelo,
            "R² (holdout)":     round(r2_score(y_test, pred), 4),
            "R² (K-Fold)":      round(scores_cv.mean(), 4),
            "Desvio K-Fold":    round(scores_cv.std(), 4),
            "RMSE (out-of-fold)": round(np.sqrt(mean_squared_error(y, pred_oof)), 2),
        })

df_ablacao = pd.DataFrame(resultados)

print("\n" + "=" * 60)
print("TESTE DE ABLAÇÃO — RESULTADO COMPLETO")
print("=" * 60)
print(df_ablacao.to_string(index=False))

# Tabela enxuta para levar ao TCC: o melhor modelo de cada configuração
print("\n--- RESUMO POR CONFIGURAÇÃO (melhor modelo de cada uma) ---")
print(
    df_ablacao.loc[df_ablacao.groupby("Configuração")["R² (K-Fold)"].idxmax()]
    .to_string(index=False)
)

r2_com    = df_ablacao[df_ablacao["Configuração"].str.startswith("A")]["R² (K-Fold)"].max()
r2_sem    = df_ablacao[df_ablacao["Configuração"].str.startswith("C")]["R² (K-Fold)"].max()
print(f"\nR² com a variável de vazamento:  {r2_com:.4f}")
print(f"R² sem a variável de vazamento:  {r2_sem:.4f}")
print(f"Queda: {r2_com - r2_sem:.4f}")
print("\nQuanto maior a queda, mais o desempenho anterior dependia do dado circular.")
print("Um R² menor porém honesto é preferível a um R² alto porém inválido.")


# %%
# PASSO 5 — Testar a transformação logarítmica da variável alvo
# Qtd_Conexoes_Total e uma variavel de CONTAGEM, muito assimetrica a direita:
# poucas capitais concentram milhares de conexoes enquanto a maioria dos
# municipios tem dezenas. Isso viola a homocedasticidade exigida pela regressao
# linear. log(1+y) comprime a cauda; usamos log1p (e nao log) porque ha
# municipios com zero conexoes, e log(0) e indefinido.
y_log = np.log1p(y)

print("\n" + "=" * 60)
print("ASSIMETRIA DA VARIÁVEL ALVO")
print("=" * 60)
print(f"Assimetria de y original: {y.skew():.3f}")
print(f"Assimetria de log(1+y):   {y_log.skew():.3f}   (mais próximo de 0 é melhor)")

comparacao_log = []
for rotulo, alvo in [("y original", y), ("log(1 + y)", y_log)]:
    scores = cross_val_score(LinearRegression(), X, alvo, cv=kfold, scoring="r2")
    comparacao_log.append({
        "Variável alvo": rotulo,
        "R² (K-Fold)":   round(scores.mean(), 4),
        "Desvio":        round(scores.std(), 4),
    })
print("\n" + pd.DataFrame(comparacao_log).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, (serie, titulo) in zip(axes, [(y, "y original"), (y_log, "log(1 + y)")]):
    ax.hist(serie, bins=60, color=COR_PONTO, alpha=0.75, edgecolor="white")
    ax.set_title(f"Distribuição — {titulo}", fontsize=12,
                 fontweight="bold", color=COR_TINTA, pad=10)
    ax.set_xlabel("Qtd Conexões", fontsize=10, color=COR_TINTA_FRACA)
    estilizar(ax)
plt.tight_layout()
plt.show()


# %%
# PASSO 6 — Verificar os pressupostos da regressão linear
# Rodado sobre o modelo final (configuracao C). Sao os testes que a banca
# espera ver para aceitar a validade estatistica do modelo.
X_sm = sm.add_constant(X)          # statsmodels exige o intercepto explicito
modelo_ols = sm.OLS(y, X_sm).fit()

print("\n" + "=" * 60)
print("SUMÁRIO DA REGRESSÃO — MODELO FINAL")
print("=" * 60)
print(modelo_ols.summary())
print("\nLeitura: a coluna P>|t| indica se cada variável é estatisticamente")
print("significativa (p < 0,05) para explicar o número de conexões.")

# --- Multicolinearidade (VIF) ---
# Mede o quanto cada variavel de entrada e explicada pelas OUTRAS entradas.
# Se duas variaveis dizem a mesma coisa, os coeficientes ficam instaveis.
#
# ATENCAO: a matriz passada a variance_inflation_factor PRECISA incluir a
# constante. Sem ela, as regressoes auxiliares rodam sem intercepto e o VIF de
# variaveis com media alta e variancia baixa (como o GHI, que so varia de 4,7 a
# 6,1) e inflado artificialmente - aqui, de 1,0 para 24,3. Versoes recentes do
# statsmodels adicionam a constante sozinhas; as antigas, nao. Passar
# explicitamente garante o mesmo resultado em qualquer versao.
X_vif = sm.add_constant(X)
colunas_vif = list(X_vif.columns)
vif = pd.DataFrame({
    "Variável": X.columns,
    "VIF": [variance_inflation_factor(X_vif.values, colunas_vif.index(c))
            for c in X.columns],
}).round(3)
print("\n--- MULTICOLINEARIDADE (VIF) ---")
print(vif.to_string(index=False))
print("Referência: VIF > 10 indica multicolinearidade problemática.")

# --- Homocedasticidade (Breusch-Pagan) ---
# Verifica se a variancia dos residuos e constante ao longo das previsoes.
_, p_bp, _, _ = het_breuschpagan(modelo_ols.resid, X_sm)
print("\n--- HOMOCEDASTICIDADE (Breusch-Pagan) ---")
print(f"p-valor: {p_bp:.6f}")
print("p < 0,05 indica heterocedasticidade — nesse caso, reportar como")
print("limitação do modelo ou adotar a versão log testada no PASSO 5.")

# --- Normalidade dos resíduos e resíduos vs. ajustados ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Q-Q plot: se os pontos seguem a diagonal, os residuos sao normais.
# Preferido ao teste de Shapiro-Wilk porque, com n = 1.793, Shapiro rejeita a
# normalidade por desvios minimos e sem relevancia pratica.
sm.qqplot(modelo_ols.resid, line="s", ax=axes[0],
          markerfacecolor=COR_PONTO, markeredgecolor="white", alpha=0.5)
axes[0].set_title("Q-Q Plot dos Resíduos", fontsize=12,
                  fontweight="bold", color=COR_TINTA, pad=10)
estilizar(axes[0])

# Residuos vs ajustados: idealmente uma nuvem sem padrao em torno do zero.
# Formato de funil = heterocedasticidade.
axes[1].scatter(modelo_ols.fittedvalues, modelo_ols.resid,
                s=22, alpha=0.4, color=COR_PONTO,
                edgecolors="white", linewidths=0.3)
axes[1].axhline(0, color=COR_TENDENCIA, linestyle="--", linewidth=2)
axes[1].set_xlabel("Valores Ajustados", fontsize=10, color=COR_TINTA_FRACA)
axes[1].set_ylabel("Resíduos", fontsize=10, color=COR_TINTA_FRACA)
axes[1].set_title("Resíduos vs. Valores Ajustados", fontsize=12,
                  fontweight="bold", color=COR_TINTA, pad=10)
estilizar(axes[1])

plt.tight_layout()
plt.show()

# --- Observações influentes (Distância de Cook) ---
# Identifica municipios que sozinhos puxam a reta de regressao. Capitais e
# regioes metropolitanas costumam aparecer aqui.
cook = modelo_ols.get_influence().cooks_distance[0]
df["cooks_d"] = cook
limite_cook = 4 / len(df)          # limite convencional

print(f"\n--- OBSERVAÇÕES INFLUENTES (Cook > {limite_cook:.5f}) ---")
print(f"Municípios acima do limite: {(cook > limite_cook).sum()}")
print("\nTop 10 mais influentes:")
print(
    df.nlargest(10, "cooks_d")[["municipio", "Estado", VAR_ALVO, "cooks_d"]]
    .round(4).to_string(index=False)
)


# %%
# PASSO 7 — Visualizar cada variável de entrada (X) contra o alvo (Y)
# Complementa o heatmap: mostra a forma da relação (linear, dispersa, com
# outliers etc.), não só um número de correlação.
# Um ponto por município, com linha de tendência e coeficiente de correlação
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, col in zip(axes, VARIAVEIS_FINAIS):
    x_vals = df[col]
    y_vals = df[VAR_ALVO]

    # Um ponto por município (contorno branco ajuda a distinguir sobreposições)
    ax.scatter(x_vals, y_vals, s=26, alpha=0.42,
               color=COR_PONTO, edgecolors="white", linewidths=0.4)

    # Ajusta uma reta (grau 1) aos pontos, so para visualizar a tendencia geral
    coef = np.polyfit(x_vals, y_vals, 1)
    x_linha = np.linspace(x_vals.min(), x_vals.max(), 100)
    ax.plot(x_linha, np.polyval(coef, x_linha), color=COR_TENDENCIA, linewidth=2)

    # Coeficiente de correlação de Pearson: -1 a 1, quanto mais perto de 1
    # (ou -1), mais forte a relação linear entre a variável e o alvo
    r = np.corrcoef(x_vals, y_vals)[0, 1]
    ax.text(0.03, 0.94, f"r = {r:.2f}", transform=ax.transAxes,
            fontsize=11, fontweight="bold", color=COR_TINTA, va="top")

    # Escala log no eixo X quando a variável é muito assimétrica
    # (poucos municípios grandes dominam a escala linear)
    if x_vals.min() > 0 and (x_vals.max() / x_vals.median()) > 20:
        ax.set_xscale("log")

    ax.set_title(ROTULOS[col], fontsize=12, fontweight="bold",
                 color=COR_TINTA, pad=10)
    ax.set_ylabel("Qtd Conexões Total", fontsize=10, color=COR_TINTA_FRACA)
    estilizar(ax)

fig.suptitle(
    "Relação entre Fatores Estruturais e Conexões Solares por Município",
    fontsize=15, fontweight="bold", color=COR_TINTA, y=1.03
)
plt.tight_layout()
plt.show()


# %%
# PASSO 8 — Modelo final e previsões fora da amostra
# Duas escolhas de modelo, cada uma com um papel:
#
#   Regressao Linear ... interpretabilidade. Os coeficientes dizem o sentido e
#                        a magnitude do efeito de cada fator (PASSO 6).
#   Random Forest ...... previsao. R² superior (0,77 vs 0,62) e, por construir
#                        a previsao como media de observacoes reais do treino,
#                        nunca devolve valor negativo - o que importa aqui,
#                        porque nao existe numero negativo de instalacoes.
#
# A Regressao Linear produzia previsao negativa em ~22% dos municipios, todos
# de baixa renda: exatamente o publico-alvo do estudo. Por isso o Random Forest
# e o modelo adotado para calcular o potencial.

modelo_linear = LinearRegression().fit(X, y)
modelo_final = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)

# IMPORTANTE: a previsao usada para calcular o potencial e FORA DA AMOSTRA
# (cross_val_predict). Um Random Forest treinado e avaliado nos mesmos dados
# praticamente memoriza o treino - os residuos ficariam artificialmente
# proximos de zero e o "potencial inexplorado" perderia o sentido. Com
# cross_val_predict, a previsao de cada municipio vem de um modelo que nao o
# viu durante o treino.
df["Qtd_Conexoes_Prevista"] = cross_val_predict(
    RandomForestRegressor(n_estimators=100, random_state=42),
    X, y, cv=kfold
).round(0)

# Potencial inexplorado: previsto - real. Positivo = o município tem menos
# conexões solares do que suas características (sol, renda, população)
# sugerem que deveria ter — ou seja, é um mercado com espaço para crescer.
# Essa leitura só é válida porque X contém apenas fatores estruturais: se a
# potência instalada estivesse em X, a previsão embutiria a própria adoção
# atual e a diferença perderia o sentido.
df["Potencial_Inexplorado"] = (
    df["Qtd_Conexoes_Prevista"] - df[VAR_ALVO]
)

# Metrica RELATIVA - normaliza pelo tamanho do municipio.
# O potencial absoluto favorece sistematicamente municipios grandes, porque o
# erro do modelo cresce com a escala (e a heterocedasticidade detectada pelo
# Breusch-Pagan). Isso faz o ranking absoluto ser dominado por capitais, que
# nao sao o objeto do estudo. Dividir pela populacao torna municipios de
# portes diferentes comparaveis e devolve o foco ao semiarido de baixa renda.
df["Potencial_por_mil_hab"] = (
    df["Potencial_Inexplorado"] / df["populacao_residente"] * 1000
).round(2)

# Taxa de realizacao: quanto da adocao esperada de fato ocorreu.
# 0,30 = o municipio tem 30% das instalacoes que seu perfil estrutural
# indicaria. Calculada so onde a previsao e positiva e nao trivial.
df["Taxa_Realizacao"] = np.where(
    df["Qtd_Conexoes_Prevista"] > 10,
    (df[VAR_ALVO] / df["Qtd_Conexoes_Prevista"]).round(3),
    np.nan
)

print("\n" + "=" * 60)
print("COEFICIENTES DA REGRESSÃO LINEAR (interpretação)")
print("=" * 60)
print(pd.DataFrame({
    "Variável":    VARIAVEIS_FINAIS,
    "Coeficiente": modelo_linear.coef_.round(4),
}).to_string(index=False))
print(f"Intercepto: {modelo_linear.intercept_:.4f}")
print("\nO intercepto não tem interpretação prática: representa a extrapolação")
print("para GHI = 0, renda = R$ 0 e população = 0, condição inexistente nos dados.")

# Coeficientes padronizados: comparaveis entre si porque estao todos na mesma
# unidade (desvios-padrao). Respondem "qual fator pesa mais?", pergunta que os
# coeficientes brutos nao respondem por estarem em unidades diferentes.
modelo_padronizado = make_pipeline(StandardScaler(), LinearRegression()).fit(X, y)
coefs_z = modelo_padronizado.named_steps["linearregression"].coef_
print("\n--- COEFICIENTES PADRONIZADOS (comparáveis entre si) ---")
print(pd.DataFrame({
    "Variável": VARIAVEIS_FINAIS,
    "Coef. padronizado": coefs_z.round(2),
}).sort_values("Coef. padronizado", key=abs, ascending=False).to_string(index=False))

# Métrica honesta para reportar: fora da amostra, via K-Fold
scores_lin   = cross_val_score(modelo_linear, X, y, cv=kfold, scoring="r2")
scores_finais = cross_val_score(modelo_final, X, y, cv=kfold, scoring="r2")
print(f"\nR² K-Fold — Regressão Linear: {scores_lin.mean():.4f} ± {scores_lin.std():.4f}")
print(f"R² K-Fold — Random Forest:    {scores_finais.mean():.4f} ± {scores_finais.std():.4f}")
print(f"\nPrevisões negativas — Linear:       {(modelo_linear.predict(X) < 0).sum()}")
print(f"Previsões negativas — Random Forest: {(df['Qtd_Conexoes_Prevista'] < 0).sum()}")


# %%
# PASSO 9 — Gráfico do modelo final (Random Forest): Real x Previsto
# Cada ponto e um municipio. Quanto mais perto da diagonal, melhor o acerto.
# As previsoes plotadas sao OUT-OF-FOLD (cross_val_predict): cada municipio foi
# previsto por um modelo que nao o viu no treino.
r2_oof   = r2_score(df[VAR_ALVO], df["Qtd_Conexoes_Prevista"])
rmse_oof = np.sqrt(mean_squared_error(df[VAR_ALVO], df["Qtd_Conexoes_Prevista"]))

# In-sample de verdade (modelo treinado e avaliado nos mesmos dados). Serve so
# para dimensionar o sobreajuste do Random Forest - NAO deve ser reportado como
# desempenho do modelo. A distancia entre este numero e o out-of-fold mostra o
# quanto o RF memoriza o treino.
r2_in_sample = r2_score(df[VAR_ALVO], modelo_final.predict(X))

print("\n" + "=" * 60)
print("DESEMPENHO DO MODELO FINAL — TRÊS MEDIDAS")
print("=" * 60)
print(f"R² in-sample (memorizado, NÃO reportar) : {r2_in_sample:.4f}")
print(f"R² out-of-fold (medida honesta)         : {r2_oof:.4f}")
print(f"R² K-Fold (média das 5 dobras)          : {scores_finais.mean():.4f} ± {scores_finais.std():.4f}")
print(f"\nRMSE out-of-fold: {rmse_oof:,.1f} conexões")
print(f"Desvio-padrão de y (baseline da média): {df[VAR_ALVO].std():,.1f} conexões")
print(f"Redução do erro sobre o baseline: {(1 - rmse_oof/df[VAR_ALVO].std()):.1%}")

fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(
    df[VAR_ALVO], df["Qtd_Conexoes_Prevista"],
    s=32, alpha=0.45,
    color=COR_PONTO,
    edgecolors="white", linewidths=0.4,
    label="Municípios"
)

limite = max(df[VAR_ALVO].max(), df["Qtd_Conexoes_Prevista"].max())
ax.plot(
    [0, limite], [0, limite],
    linestyle="--", linewidth=2, color=COR_TENDENCIA,
    label="Previsão perfeita"
)

# Ambas as metricas exibidas sao fora da amostra - nenhum municipio do grafico
# foi previsto por um modelo que o tinha visto no treino.
ax.text(
    0.03, 0.94,
    f"Random Forest — previsões out-of-fold\n"
    f"R² (K-Fold) = {scores_finais.mean():.3f}\n"
    f"R² (out-of-fold) = {r2_oof:.3f}\n"
    f"RMSE = {rmse_oof:,.0f} conexões",
    transform=ax.transAxes,
    fontsize=11, fontweight="bold", color=COR_TINTA,
    va="top"
)

ax.set_xlabel("Qtd Conexões Real", fontsize=11, color=COR_TINTA_FRACA)
ax.set_ylabel("Qtd Conexões Prevista", fontsize=11, color=COR_TINTA_FRACA)
ax.set_title(
    "Modelo Final — Conexões Solares por Município (Real vs Previsto)",
    fontsize=13, fontweight="bold", color=COR_TINTA, pad=12
)
estilizar(ax)
ax.legend(loc="lower right", frameon=False, fontsize=10)
plt.tight_layout()
plt.show()


# %%
# PASSO 10 — Rankings de potencial inexplorado
# Saida que alimenta a secao de recomendacoes para politicas publicas do TCC.
# Apresentamos DOIS rankings porque eles respondem a perguntas diferentes -
# e apenas o segundo responde a pergunta do trabalho.

COLUNAS_RANKING = [
    "municipio", "Estado", "populacao_residente", "ghi_media_kwh_m2_dia",
    "renda_per_capita", VAR_ALVO, "Qtd_Conexoes_Prevista",
    "Potencial_Inexplorado", "Potencial_por_mil_hab",
]

# --- Ranking 1: absoluto (volume de mercado) ---
# Responde "onde há mais instalações a ganhar em números absolutos?".
# Tende a ser dominado por capitais e regioes metropolitanas - util para o
# setor produtivo, mas nao para focalizar politica social.
print("\n" + "=" * 60)
print("RANKING 1 — POTENCIAL ABSOLUTO (volume de mercado)")
print("=" * 60)
print("Pergunta: onde há mais instalações a ganhar, em números absolutos?")
print(df.nlargest(15, "Potencial_Inexplorado")[COLUNAS_RANKING].to_string(index=False))

# --- Ranking 2: relativo (alinhado a pergunta do TCC) ---
# Responde "onde a adocao esta mais aquem do que o perfil do municipio
# indicaria, proporcionalmente ao seu tamanho?". Ao normalizar pela populacao,
# municipios pequenos do semiarido deixam de ser ofuscados pelas capitais.
# Filtro de populacao minima evita que municipios minusculos, onde poucas
# instalacoes geram razoes extremas, dominem o topo por ruido.
POP_MINIMA = 5000
elegiveis = df[df["populacao_residente"] >= POP_MINIMA]

print("\n" + "=" * 60)
print("RANKING 2 — POTENCIAL RELATIVO (por mil habitantes)")
print("=" * 60)
print(f"Pergunta: onde a adoção está proporcionalmente mais aquém do esperado?")
print(f"(municípios com população ≥ {POP_MINIMA:,} habitantes)")
print(elegiveis.nlargest(20, "Potencial_por_mil_hab")[COLUNAS_RANKING].to_string(index=False))

# --- Perfil comparado dos dois rankings ---
# Evidencia, com numeros, por que o ranking relativo e o adequado ao estudo.
top_abs = df.nlargest(20, "Potencial_Inexplorado")
top_rel = elegiveis.nlargest(20, "Potencial_por_mil_hab")

print("\n--- PERFIL DOS DOIS RANKINGS (mediana) ---")
print(pd.DataFrame({
    "Indicador": ["População", "Renda per capita (R$)", "GHI (kWh/m²/dia)"],
    "Top 20 absoluto": [
        f"{top_abs['populacao_residente'].median():,.0f}",
        f"{top_abs['renda_per_capita'].median():,.2f}",
        f"{top_abs['ghi_media_kwh_m2_dia'].median():.3f}",
    ],
    "Top 20 relativo": [
        f"{top_rel['populacao_residente'].median():,.0f}",
        f"{top_rel['renda_per_capita'].median():,.2f}",
        f"{top_rel['ghi_media_kwh_m2_dia'].median():.3f}",
    ],
    "Dataset": [
        f"{df['populacao_residente'].median():,.0f}",
        f"{df['renda_per_capita'].median():,.2f}",
        f"{df['ghi_media_kwh_m2_dia'].median():.3f}",
    ],
}).to_string(index=False))

# --- Recorte prioritario: alta irradiacao + baixa renda + adocao aquem ---
# Interseccao que traduz diretamente o objetivo do TCC: municipios que reunem
# excelente recurso solar, populacao de baixa renda e adocao abaixo do
# esperado. Sao os candidatos naturais a programas de subsidio.
ghi_alto    = df["ghi_media_kwh_m2_dia"] >= df["ghi_media_kwh_m2_dia"].quantile(0.75)
renda_baixa = df["renda_per_capita"]     <= df["renda_per_capita"].quantile(0.50)
defasado    = df["Potencial_por_mil_hab"] > 0
porte_min   = df["populacao_residente"]  >= POP_MINIMA

prioritarios = df[ghi_alto & renda_baixa & defasado & porte_min]

print("\n" + "=" * 60)
print("RECORTE PRIORITÁRIO PARA POLÍTICA PÚBLICA")
print("=" * 60)
print("Critérios: GHI no quartil superior + renda na metade inferior +")
print(f"adoção abaixo do esperado + população ≥ {POP_MINIMA:,}")
print(f"\nMunicípios que atendem aos quatro critérios: {len(prioritarios)}")
print(f"Distribuição por estado:")
print(prioritarios["Estado"].value_counts().to_string())
print("\nTop 20:")
print(prioritarios.nlargest(20, "Potencial_por_mil_hab")[COLUNAS_RANKING].to_string(index=False))


# %%
# PASSO 11 — Exportar os resultados finais para Excel
PASTA_DADOS = os.path.join("..", "dados")

df.to_excel(os.path.join(PASTA_DADOS, "dados.xlsx"), index=False)
print(f"\nDataset com previsões exportado: {os.path.join(PASTA_DADOS, 'dados.xlsx')}")

# Tabela do teste de ablação — vai direto para a seção de Resultados do TCC
df_ablacao.to_excel(os.path.join(PASTA_DADOS, "teste_ablacao.xlsx"), index=False)
print(f"Tabela do teste de ablação exportada: {os.path.join(PASTA_DADOS, 'teste_ablacao.xlsx')}")

# Recorte prioritario em arquivo proprio - e a tabela que sustenta a secao de
# recomendacoes. Mantido separado para nao ser perdido quando dados.xlsx for
# sobrescrito na proxima execucao.
prioritarios.nlargest(50, "Potencial_por_mil_hab")[COLUNAS_RANKING].to_excel(
    os.path.join(PASTA_DADOS, "municipios_prioritarios.xlsx"), index=False
)
print(f"Recorte prioritário exportado: {os.path.join(PASTA_DADOS, 'municipios_prioritarios.xlsx')}")
