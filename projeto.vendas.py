from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import ollama
import re
import unicodedata
from difflib import get_close_matches

# ============================================================
# CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="Game Sales Analytics",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# ESTILO
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    [data-testid="stMetric"] {
        background-color: #f7f7f7;
        border-radius: 12px;
        padding: 15px;
        border: 1px solid #e5e5e5;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CAMINHO DO DATASET
# ============================================================

PASTA_PROJETO = Path(__file__).resolve().parent
CAMINHO_CSV = PASTA_PROJETO / "vgsales.csv"


# ============================================================
# CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ============================================================

@st.cache_data
def carregar_dados(caminho):
    """
    Carrega e prepara o dataset de vendas de jogos.
    """

    df = pd.read_csv(caminho)

    # --------------------------------------------------------
    # Limpeza dos nomes das colunas
    # --------------------------------------------------------

    df.columns = df.columns.str.strip()

    # --------------------------------------------------------
    # Limpeza de textos
    # --------------------------------------------------------

    colunas_texto = [
        "Name",
        "Platform",
        "Genre",
        "Publisher"
    ]

    for coluna in colunas_texto:
        if coluna in df.columns:
            df[coluna] = df[coluna].astype("string").str.strip()

    # --------------------------------------------------------
    # Valores ausentes
    # --------------------------------------------------------

    if "Publisher" in df.columns:
        df["Publisher"] = df["Publisher"].fillna("Desconhecido")

    if "Genre" in df.columns:
        df["Genre"] = df["Genre"].fillna("Desconhecido")

    if "Platform" in df.columns:
        df["Platform"] = df["Platform"].fillna("Desconhecida")

    # --------------------------------------------------------
    # Conversão numérica
    # --------------------------------------------------------

    colunas_numericas = [
        "Rank",
        "Year",
        "NA_Sales",
        "EU_Sales",
        "JP_Sales",
        "Other_Sales",
        "Global_Sales"
    ]

    for coluna in colunas_numericas:
        if coluna in df.columns:
            df[coluna] = pd.to_numeric(
                df[coluna],
                errors="coerce"
            )

    return df


# ============================================================
# VERIFICAÇÃO DO DATASET
# ============================================================

if not CAMINHO_CSV.exists():

    st.error(
        "❌ O arquivo vgsales.csv não foi encontrado."
    )

    st.info(
        "Coloque o arquivo vgsales.csv na mesma pasta deste "
        "arquivo Python."
    )

    st.code(str(CAMINHO_CSV))

    st.stop()


try:

    df = carregar_dados(str(CAMINHO_CSV))

except Exception as erro:

    st.error(
        f"❌ Não foi possível carregar o dataset: {erro}"
    )

    st.stop()


# ============================================================
# VALIDAÇÃO DAS COLUNAS
# ============================================================

COLUNAS_OBRIGATORIAS = [
    "Rank",
    "Name",
    "Platform",
    "Year",
    "Genre",
    "Publisher",
    "NA_Sales",
    "EU_Sales",
    "JP_Sales",
    "Other_Sales",
    "Global_Sales"
]

colunas_faltantes = [
    coluna
    for coluna in COLUNAS_OBRIGATORIAS
    if coluna not in df.columns
]

if colunas_faltantes:

    st.error(
        "❌ O dataset não possui todas as colunas esperadas."
    )

    st.write(
        "Colunas faltantes:",
        colunas_faltantes
    )

    st.stop()


# ============================================================
# FUNÇÕES DE ANÁLISE
# ============================================================

def vendas_por_plataforma(dataframe, limite=None):

    resultado = (
        dataframe
        .groupby("Platform", dropna=False)["Global_Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    if limite is not None:
        resultado = resultado.head(limite)

    return resultado


def vendas_por_genero(dataframe, limite=None):

    resultado = (
        dataframe
        .groupby("Genre", dropna=False)["Global_Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    if limite is not None:
        resultado = resultado.head(limite)

    return resultado


def vendas_por_publisher(dataframe, limite=None):

    resultado = (
        dataframe
        .groupby("Publisher", dropna=False)["Global_Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    if limite is not None:
        resultado = resultado.head(limite)

    return resultado


def top_jogos(dataframe, limite=10):

    return (
        dataframe[
            [
                "Name",
                "Platform",
                "Year",
                "Genre",
                "Publisher",
                "Global_Sales"
            ]
        ]
        .sort_values(
            "Global_Sales",
            ascending=False
        )
        .head(limite)
    )


def vendas_por_regiao(dataframe):

    return pd.Series(
        {
            "América do Norte": dataframe["NA_Sales"].sum(),
            "Europa": dataframe["EU_Sales"].sum(),
            "Japão": dataframe["JP_Sales"].sum(),
            "Outras": dataframe["Other_Sales"].sum()
        }
    )


def quantidade_por_genero(dataframe):

    return (
        dataframe["Genre"]
        .value_counts()
    )


def vendas_por_ano(dataframe):

    return (
        dataframe
        .dropna(subset=["Year"])
        .groupby("Year")["Global_Sales"]
        .sum()
        .reset_index()
        .sort_values("Year")
    )


# ============================================================
# CONTEXTO PARA A IA
# ============================================================

def gerar_contexto_ia(dataframe, pergunta):

    """
    Gera um contexto completo usando Pandas para a IA.

    O Qwen não calcula os dados.
    O Python/Pandas calcula e o Qwen interpreta os resultados.
    """

    if dataframe.empty:
        return """
O filtro atual não possui nenhum jogo.

Não existem dados disponíveis para realizar a análise.
"""

    # ========================================================
    # RESUMO GERAL
    # ========================================================

    total_jogos = len(dataframe)

    vendas_totais = dataframe["Global_Sales"].sum()

    vendas_media = dataframe["Global_Sales"].mean()

    vendas_mediana = dataframe["Global_Sales"].median()

    maior_venda = dataframe["Global_Sales"].max()

    menor_venda = dataframe["Global_Sales"].min()


    # ========================================================
    # TOP JOGOS
    # ========================================================

    top_jogos_df = (
        dataframe[
            [
                "Name",
                "Platform",
                "Year",
                "Genre",
                "Publisher",
                "Global_Sales"
            ]
        ]
        .sort_values(
            "Global_Sales",
            ascending=False
        )
        .head(10)
    )


    # ========================================================
    # PLATAFORMAS
    # ========================================================

    vendas_plataforma = (
        dataframe
        .groupby("Platform")["Global_Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    media_plataforma = (
        dataframe
        .groupby("Platform")["Global_Sales"]
        .mean()
        .sort_values(ascending=False)
    )

    quantidade_plataforma = (
        dataframe["Platform"]
        .value_counts()
    )


    # ========================================================
    # GÊNEROS
    # ========================================================

    vendas_genero = (
        dataframe
        .groupby("Genre")["Global_Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    media_genero = (
        dataframe
        .groupby("Genre")["Global_Sales"]
        .mean()
        .sort_values(ascending=False)
    )

    quantidade_genero = (
        dataframe["Genre"]
        .value_counts()
    )


    # ========================================================
    # PUBLISHERS
    # ========================================================

    vendas_publisher = (
        dataframe
        .groupby("Publisher")["Global_Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    media_publisher = (
        dataframe
        .groupby("Publisher")["Global_Sales"]
        .mean()
        .sort_values(ascending=False)
    )

    quantidade_publisher = (
        dataframe["Publisher"]
        .value_counts()
    )


    # ========================================================
    # REGIÕES
    # ========================================================

    vendas_regiao = {
        "América do Norte":
            dataframe["NA_Sales"].sum(),

        "Europa":
            dataframe["EU_Sales"].sum(),

        "Japão":
            dataframe["JP_Sales"].sum(),

        "Outras":
            dataframe["Other_Sales"].sum()
    }


    # ========================================================
    # PARTICIPAÇÃO REGIONAL
    # ========================================================

    participacao_regional = {}

    for regiao, vendas in vendas_regiao.items():

        if vendas_totais > 0:

            participacao_regional[regiao] = (
                vendas / vendas_totais
            ) * 100

        else:

            participacao_regional[regiao] = 0


    # ========================================================
    # VENDAS POR ANO
    # ========================================================

    vendas_ano = (
        dataframe
        .dropna(subset=["Year"])
        .groupby("Year")["Global_Sales"]
        .sum()
        .sort_values(ascending=False)
    )


    # ========================================================
    # CONSTRUÇÃO DO CONTEXTO
    # ========================================================

    contexto = f"""
============================================================
DADOS REAIS DO DATASET
============================================================

Todos os números abaixo foram calculados pelo Python/Pandas.

FILTRO ATUAL
------------

Quantidade de jogos:
{total_jogos}

Vendas globais totais:
{vendas_totais:.2f} milhões

Média de vendas por jogo:
{vendas_media:.2f} milhões

Mediana de vendas por jogo:
{vendas_mediana:.2f} milhões

Maior venda individual:
{maior_venda:.2f} milhões

Menor venda:
{menor_venda:.2f} milhões


============================================================
TOP 10 JOGOS
============================================================

"""


    for _, jogo in top_jogos_df.iterrows():

        contexto += (
            f"- {jogo['Name']} | "
            f"Plataforma: {jogo['Platform']} | "
            f"Ano: {jogo['Year']} | "
            f"Gênero: {jogo['Genre']} | "
            f"Publisher: {jogo['Publisher']} | "
            f"Vendas: "
            f"{jogo['Global_Sales']:.2f} milhões\n"
        )


    contexto += """

============================================================
TOP PLATAFORMAS POR VENDAS
============================================================

"""


    for plataforma, vendas in vendas_plataforma.head(10).items():

        quantidade = quantidade_plataforma.get(
            plataforma,
            0
        )

        media = media_plataforma.get(
            plataforma,
            0
        )

        contexto += (
            f"- {plataforma}: "
            f"{vendas:.2f} milhões | "
            f"{quantidade} jogos | "
            f"Média: {media:.2f} milhões por jogo\n"
        )


    contexto += """

============================================================
TOP GÊNEROS POR VENDAS
============================================================

"""


    for genero, vendas in vendas_genero.items():

        quantidade = quantidade_genero.get(
            genero,
            0
        )

        media = media_genero.get(
            genero,
            0
        )

        contexto += (
            f"- {genero}: "
            f"{vendas:.2f} milhões | "
            f"{quantidade} jogos | "
            f"Média: {media:.2f} milhões por jogo\n"
        )


    contexto += """

============================================================
TOP PUBLISHERS POR VENDAS
============================================================

"""


    for publisher, vendas in vendas_publisher.head(10).items():

        quantidade = quantidade_publisher.get(
            publisher,
            0
        )

        media = media_publisher.get(
            publisher,
            0
        )

        contexto += (
            f"- {publisher}: "
            f"{vendas:.2f} milhões | "
            f"{quantidade} jogos | "
            f"Média: {media:.2f} milhões por jogo\n"
        )


    contexto += """

============================================================
VENDAS POR REGIÃO
============================================================

"""


    for regiao, vendas in vendas_regiao.items():

        participacao = participacao_regional[regiao]

        contexto += (
            f"- {regiao}: "
            f"{vendas:.2f} milhões | "
            f"Participação: {participacao:.2f}%\n"
        )


    contexto += """

============================================================
VENDAS POR ANO
============================================================

"""


    for ano, vendas in vendas_ano.head(15).items():

        contexto += (
            f"- {int(ano)}: "
            f"{vendas:.2f} milhões\n"
        )


    # ========================================================
    # MELHORES RESULTADOS
    # ========================================================

    if not vendas_plataforma.empty:

        contexto += f"""

============================================================
DESTAQUES
============================================================

Plataforma com maior volume:
{vendas_plataforma.index[0]}

Vendas da plataforma líder:
{vendas_plataforma.iloc[0]:.2f} milhões

Plataforma com maior média por jogo:
{media_plataforma.index[0]}

Média da plataforma:
{media_plataforma.iloc[0]:.2f} milhões
"""


    if not vendas_genero.empty:

        contexto += f"""

Gênero com maior volume:
{vendas_genero.index[0]}

Vendas do gênero líder:
{vendas_genero.iloc[0]:.2f} milhões

Gênero com maior média por jogo:
{media_genero.index[0]}

Média:
{media_genero.iloc[0]:.2f} milhões
"""


    if not vendas_publisher.empty:

        contexto += f"""

Publisher com maior volume:
{vendas_publisher.index[0]}

Vendas:
{vendas_publisher.iloc[0]:.2f} milhões

Publisher com maior média por jogo:
{media_publisher.index[0]}

Média:
{media_publisher.iloc[0]:.2f} milhões
"""


    # ========================================================
    # BUSCA POR NOME DE JOGO
    # ========================================================

    pergunta_limpa = (
        str(pergunta)
        .strip()
        .lower()
    )


    if len(pergunta_limpa) >= 3:

        encontrados = dataframe[
            dataframe["Name"]
            .str.lower()
            .str.contains(
                pergunta_limpa,
                na=False,
                regex=False
            )
        ]


        if not encontrados.empty:

            contexto += """

============================================================
JOGOS ENCONTRADOS RELACIONADOS À PERGUNTA
============================================================

"""


            for _, jogo in encontrados.head(10).iterrows():

                contexto += (
                    f"- {jogo['Name']} | "
                    f"Plataforma: {jogo['Platform']} | "
                    f"Ano: {jogo['Year']} | "
                    f"Gênero: {jogo['Genre']} | "
                    f"Publisher: {jogo['Publisher']} | "
                    f"América do Norte: "
                    f"{jogo['NA_Sales']:.2f} | "
                    f"Europa: "
                    f"{jogo['EU_Sales']:.2f} | "
                    f"Japão: "
                    f"{jogo['JP_Sales']:.2f} | "
                    f"Outras: "
                    f"{jogo['Other_Sales']:.2f} | "
                    f"Global: "
                    f"{jogo['Global_Sales']:.2f} milhões\n"
                )


    return contexto

    # --------------------------------------------------------
    # Contexto básico
    # --------------------------------------------------------

    contexto = f"""
DADOS CALCULADOS PELO PYTHON
============================

Quantidade de jogos no filtro atual:
{total_jogos}

Vendas globais totais:
{vendas_totais:.2f} milhões

TOP PLATAFORMAS POR VENDAS
---------------------------

"""

    for plataforma, vendas in plataformas.items():

        contexto += (
            f"- {plataforma}: "
            f"{vendas:.2f} milhões\n"
        )


    contexto += """

TOP GÊNEROS POR VENDAS
----------------------

"""

    for genero, vendas in generos.items():

        contexto += (
            f"- {genero}: "
            f"{vendas:.2f} milhões\n"
        )


    contexto += """

TOP PUBLISHERS POR VENDAS
-------------------------

"""

    for publisher, vendas in publishers.items():

        contexto += (
            f"- {publisher}: "
            f"{vendas:.2f} milhões\n"
        )


    contexto += """

TOP JOGOS
---------

"""

    for _, jogo in jogos.iterrows():

        contexto += (
            f"- {jogo['Name']} | "
            f"Plataforma: {jogo['Platform']} | "
            f"Ano: {jogo['Year']} | "
            f"Gênero: {jogo['Genre']} | "
            f"Vendas: {jogo['Global_Sales']:.2f} milhões\n"
        )


    contexto += """

VENDAS POR REGIÃO
-----------------

"""

    for regiao, vendas in regioes.items():

        contexto += (
            f"- {regiao}: "
            f"{vendas:.2f} milhões\n"
        )


    contexto += """

QUANTIDADE DE JOGOS POR GÊNERO
------------------------------

"""

    for genero, quantidade in quantidades_genero.items():

        contexto += (
            f"- {genero}: "
            f"{quantidade} jogos\n"
        )


    # --------------------------------------------------------
    # Procurar jogos relacionados à pergunta
    # --------------------------------------------------------

    pergunta_limpa = str(pergunta).strip().lower()

    if len(pergunta_limpa) >= 3:

        encontrados = dataframe[
            dataframe["Name"]
            .str.lower()
            .str.contains(
                pergunta_limpa,
                na=False,
                regex=False
            )
        ]

        if not encontrados.empty:

            contexto += """

JOGOS ENCONTRADOS PELO NOME DA PERGUNTA
---------------------------------------

"""

            for _, jogo in encontrados.head(10).iterrows():

                contexto += (
                    f"- {jogo['Name']} | "
                    f"Plataforma: {jogo['Platform']} | "
                    f"Ano: {jogo['Year']} | "
                    f"Gênero: {jogo['Genre']} | "
                    f"Publisher: {jogo['Publisher']} | "
                    f"Vendas globais: "
                    f"{jogo['Global_Sales']:.2f} milhões\n"
                )


    return contexto


# ============================================================
# TÍTULO
# ============================================================

st.title("🎮 Game Sales Analytics")

st.markdown(
    """
    **Dashboard interativo de análise de vendas de videogames**

    Explore jogos, plataformas, gêneros, publishers, regiões
    e evolução das vendas ao longo dos anos.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🎛️ Filtros")


# ------------------------------------------------------------
# ANO
# ------------------------------------------------------------

anos = sorted(
    df["Year"]
    .dropna()
    .unique()
)

if anos:

    ano_min = int(min(anos))
    ano_max = int(max(anos))

    ano_selecionado = st.sidebar.slider(
        "Período",
        min_value=ano_min,
        max_value=ano_max,
        value=(ano_min, ano_max)
    )

else:

    ano_selecionado = None


# ------------------------------------------------------------
# PLATAFORMAS
# ------------------------------------------------------------

plataformas = sorted(
    df["Platform"]
    .dropna()
    .unique()
)

plataformas_selecionadas = st.sidebar.multiselect(
    "Plataformas",
    plataformas,
    default=plataformas
)


# ------------------------------------------------------------
# GÊNEROS
# ------------------------------------------------------------

generos = sorted(
    df["Genre"]
    .dropna()
    .unique()
)

generos_selecionados = st.sidebar.multiselect(
    "Gêneros",
    generos,
    default=generos
)


# ============================================================
# FILTRO PRINCIPAL
# ============================================================

df_filtrado = df.copy()


if ano_selecionado is not None:

    df_filtrado = df_filtrado[
        (
            df_filtrado["Year"]
            .between(
                ano_selecionado[0],
                ano_selecionado[1]
            )
        )
        |
        (
            df_filtrado["Year"].isna()
        )
    ]


if plataformas_selecionadas:

    df_filtrado = df_filtrado[
        df_filtrado["Platform"]
        .isin(plataformas_selecionadas)
    ]

else:

    df_filtrado = df_filtrado.iloc[0:0]


if generos_selecionados:

    df_filtrado = df_filtrado[
        df_filtrado["Genre"]
        .isin(generos_selecionados)
    ]

else:

    df_filtrado = df_filtrado.iloc[0:0]


# ============================================================
# AVISO FILTRO VAZIO
# ============================================================

if df_filtrado.empty:

    st.warning(
        "⚠️ Nenhum jogo corresponde aos filtros selecionados."
    )

    st.info(
        "Selecione pelo menos uma plataforma e um gênero."
    )


# ============================================================
# KPIs
# ============================================================

total_jogos = len(df_filtrado)

vendas_globais = df_filtrado["Global_Sales"].sum()

plataforma_top = vendas_por_plataforma(
    df_filtrado
)

genero_top = vendas_por_genero(
    df_filtrado
)

publisher_top = vendas_por_publisher(
    df_filtrado
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "🎮 Jogos",
        f"{total_jogos:,}".replace(",", ".")
    )


with col2:

    valor_formatado = (
        f"{vendas_globais:,.2f}"
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )

    st.metric(
        "🌎 Vendas globais",
        f"{valor_formatado} M"
    )


with col3:

    if not plataforma_top.empty:

        st.metric(
            "🕹️ Plataforma líder",
            plataforma_top.index[0]
        )

    else:

        st.metric(
            "🕹️ Plataforma líder",
            "-"
        )


with col4:

    if not genero_top.empty:

        st.metric(
            "🎯 Gênero líder",
            genero_top.index[0]
        )

    else:

        st.metric(
            "🎯 Gênero líder",
            "-"
        )


st.divider()


# ============================================================
# ABAS
# ============================================================

aba1, aba2, aba3, aba4, aba5, aba6 = st.tabs(
    [
        "📊 Visão geral",
        "🕹️ Plataformas",
        "🎯 Gêneros",
        "🏢 Publishers",
        "🌎 Regiões",
        "💡 Insights"
    ]
)


# ============================================================
# ABA 1 — VISÃO GERAL
# ============================================================

with aba1:

    st.subheader("📊 Visão geral das vendas")


    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # TOP JOGOS
    # --------------------------------------------------------

    with col1:

        st.markdown("### 🏆 Top 10 jogos")

        top = top_jogos(
            df_filtrado,
            limite=10
        )

        if not top.empty:

            fig = px.bar(
                top.sort_values(
                    "Global_Sales"
                ),
                x="Global_Sales",
                y="Name",
                orientation="h",
                title="Top 10 jogos por vendas globais",
                labels={
                    "Global_Sales":
                        "Vendas globais (milhões)",
                    "Name":
                        "Jogo"
                },
                hover_data=[
                    "Platform",
                    "Year",
                    "Genre"
                ]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info(
                "Não há dados para exibir."
            )


    # --------------------------------------------------------
    # REGIÕES
    # --------------------------------------------------------

    with col2:

        st.markdown("### 🌎 Vendas por região")

        regioes = vendas_por_regiao(
            df_filtrado
        )

        regioes_df = (
            regioes
            .rename("Vendas")
            .reset_index()
        )

        regioes_df.columns = [
            "Região",
            "Vendas"
        ]

        if not regioes_df.empty:

            fig = px.bar(
                regioes_df.sort_values(
                    "Vendas"
                ),
                x="Vendas",
                y="Região",
                orientation="h",
                title="Vendas por região",
                labels={
                    "Vendas":
                        "Vendas (milhões)"
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


    # --------------------------------------------------------
    # EVOLUÇÃO
    # --------------------------------------------------------

    st.markdown(
        "### 📈 Evolução das vendas ao longo dos anos"
    )

    vendas_ano = vendas_por_ano(
        df_filtrado
    )

    if not vendas_ano.empty:

        fig = px.line(
            vendas_ano,
            x="Year",
            y="Global_Sales",
            markers=True,
            title="Vendas globais por ano",
            labels={
                "Year": "Ano",
                "Global_Sales":
                    "Vendas globais (milhões)"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "Não há anos válidos para exibir."
        )


# ============================================================
# ABA 2 — PLATAFORMAS
# ============================================================

with aba2:

    st.subheader("🕹️ Análise de plataformas")

    vendas_plataforma = vendas_por_plataforma(
        df_filtrado
    )

    col1, col2 = st.columns(2)


    with col1:

        if not vendas_plataforma.empty:

            dados_grafico = (
                vendas_plataforma
                .head(10)
                .sort_values()
                .rename("Vendas")
                .reset_index()
            )

            fig = px.bar(
                dados_grafico,
                x="Vendas",
                y="Platform",
                orientation="h",
                title="Top 10 plataformas por vendas",
                labels={
                    "Vendas":
                        "Vendas globais (milhões)",
                    "Platform":
                        "Plataforma"
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info(
                "Não há plataformas para exibir."
            )


    with col2:

        if not vendas_plataforma.empty:

            plataforma_escolhida = st.selectbox(
                "Escolha uma plataforma",
                vendas_plataforma.index.tolist()
            )

            dados_plataforma = df_filtrado[
                df_filtrado["Platform"]
                == plataforma_escolhida
            ]

            vendas_ano_plataforma = (
                dados_plataforma
                .dropna(subset=["Year"])
                .groupby("Year")["Global_Sales"]
                .sum()
                .reset_index()
                .sort_values("Year")
            )

            if not vendas_ano_plataforma.empty:

                fig = px.line(
                    vendas_ano_plataforma,
                    x="Year",
                    y="Global_Sales",
                    markers=True,
                    title=(
                        "Evolução das vendas — "
                        f"{plataforma_escolhida}"
                    ),
                    labels={
                        "Year": "Ano",
                        "Global_Sales":
                            "Vendas (milhões)"
                    }
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:

                st.info(
                    "Não existem anos válidos "
                    "para essa plataforma."
                )


# ============================================================
# ABA 3 — GÊNEROS
# ============================================================

with aba3:

    st.subheader("🎯 Análise de gêneros")

    vendas_genero = vendas_por_genero(
        df_filtrado
    )

    quantidade_genero = quantidade_por_genero(
        df_filtrado
    )


    col1, col2 = st.columns(2)


    with col1:

        if not vendas_genero.empty:

            dados_grafico = (
                vendas_genero
                .sort_values()
                .rename("Vendas")
                .reset_index()
            )

            fig = px.bar(
                dados_grafico,
                x="Vendas",
                y="Genre",
                orientation="h",
                title="Vendas globais por gênero",
                labels={
                    "Vendas":
                        "Vendas (milhões)",
                    "Genre":
                        "Gênero"
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


    with col2:

        if not quantidade_genero.empty:

            dados_grafico = (
                quantidade_genero
                .sort_values()
                .rename("Quantidade")
                .reset_index()
            )

            fig = px.bar(
                dados_grafico,
                x="Quantidade",
                y="Genre",
                orientation="h",
                title="Quantidade de jogos por gênero",
                labels={
                    "Quantidade":
                        "Quantidade de jogos",
                    "Genre":
                        "Gênero"
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


    # --------------------------------------------------------
    # MÉDIA
    # --------------------------------------------------------

    st.markdown(
        "### 🔎 Vendas médias por jogo"
    )

    media_genero = (
        df_filtrado
        .groupby("Genre")["Global_Sales"]
        .mean()
        .sort_values(ascending=False)
    )

    if not media_genero.empty:

        dados_grafico = (
            media_genero
            .rename("Media")
            .reset_index()
        )

        fig = px.bar(
            dados_grafico,
            x="Genre",
            y="Media",
            title="Média de vendas por jogo — por gênero",
            labels={
                "Media":
                    "Vendas médias (milhões)",
                "Genre":
                    "Gênero"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# ABA 4 — PUBLISHERS
# ============================================================

with aba4:

    st.subheader("🏢 Análise de publishers")


    quantidade_publisher = (
        df_filtrado["Publisher"]
        .value_counts()
        .head(10)
    )

    vendas_publisher = vendas_por_publisher(
        df_filtrado,
        limite=10
    )


    col1, col2 = st.columns(2)


    with col1:

        if not quantidade_publisher.empty:

            dados_grafico = (
                quantidade_publisher
                .sort_values()
                .rename("Quantidade")
                .reset_index()
            )

            fig = px.bar(
                dados_grafico,
                x="Quantidade",
                y="Publisher",
                orientation="h",
                title=(
                    "Top 10 publishers "
                    "por quantidade de jogos"
                ),
                labels={
                    "Quantidade":
                        "Quantidade de jogos",
                    "Publisher":
                        "Publisher"
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


    with col2:

        if not vendas_publisher.empty:

            dados_grafico = (
                vendas_publisher
                .sort_values()
                .rename("Vendas")
                .reset_index()
            )

            fig = px.bar(
                dados_grafico,
                x="Vendas",
                y="Publisher",
                orientation="h",
                title=(
                    "Top 10 publishers "
                    "por vendas"
                ),
                labels={
                    "Vendas":
                        "Vendas globais (milhões)",
                    "Publisher":
                        "Publisher"
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# ABA 5 — REGIÕES
# ============================================================

with aba5:

    st.subheader("🌎 Análise regional")


    regioes = vendas_por_regiao(
        df_filtrado
    )

    regioes_df = (
        regioes
        .rename("Vendas")
        .reset_index()
    )

    regioes_df.columns = [
        "Região",
        "Vendas"
    ]


    if not regioes_df.empty:

        fig = px.bar(
            regioes_df.sort_values(
                "Vendas"
            ),
            x="Vendas",
            y="Região",
            orientation="h",
            title="Distribuição das vendas por região",
            labels={
                "Vendas":
                    "Vendas (milhões)"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # GÊNEROS POR REGIÃO
    # --------------------------------------------------------

    st.markdown(
        "### 🎯 Gêneros por região"
    )

    genero_regiao = (
        df_filtrado
        .groupby("Genre")[
            [
                "NA_Sales",
                "EU_Sales",
                "JP_Sales",
                "Other_Sales"
            ]
        ]
        .sum()
        .sort_values(
            "NA_Sales",
            ascending=False
        )
        .head(10)
        .reset_index()
    )


    if not genero_regiao.empty:

        fig = px.bar(
            genero_regiao,
            x="Genre",
            y=[
                "NA_Sales",
                "EU_Sales",
                "JP_Sales",
                "Other_Sales"
            ],
            barmode="group",
            title="Top gêneros por região",
            labels={
                "value":
                    "Vendas (milhões)",
                "Genre":
                    "Gênero",
                "variable":
                    "Região"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# ABA 6 — INSIGHTS
# ============================================================

with aba6:

    st.subheader("💡 Insights automáticos")


    if not genero_top.empty:

        genero_maior = genero_top.index[0]

        vendas_genero_maior = (
            genero_top.iloc[0]
        )

        st.markdown(
            "### 🎯 Gênero com maior volume"
        )

        st.info(
            f"No filtro atual, **{genero_maior}** "
            f"possui o maior volume de vendas globais, "
            f"com **{vendas_genero_maior:.2f} milhões**."
        )


    if not plataforma_top.empty:

        st.markdown(
            "### 🕹️ Plataforma com maior volume"
        )

        st.info(
            f"No filtro atual, **{plataforma_top.index[0]}** "
            f"possui o maior volume de vendas, "
            f"com **{plataforma_top.iloc[0]:.2f} milhões**."
        )


    if not publisher_top.empty:

        st.markdown(
            "### 🏢 Publisher com maior volume"
        )

        st.info(
            f"No filtro atual, **{publisher_top.index[0]}** "
            f"possui o maior volume de vendas, "
            f"com **{publisher_top.iloc[0]:.2f} milhões**."
        )


    # --------------------------------------------------------
    # MÉDIA POR GÊNERO
    # --------------------------------------------------------

    media_genero = (
        df_filtrado
        .groupby("Genre")["Global_Sales"]
        .mean()
        .sort_values(ascending=False)
    )

    if not media_genero.empty:

        st.markdown(
            "### 📊 Média de vendas por jogo"
        )

        st.info(
            f"O gênero com maior média de vendas "
            f"por título no filtro atual é "
            f"**{media_genero.index[0]}**, "
            f"com média de "
            f"**{media_genero.iloc[0]:.2f} milhões** "
            f"por jogo."
        )


# ============================================================
# TABELA DETALHADA
# ============================================================

st.divider()

st.subheader("🔎 Explorar jogos")


col1, col2 = st.columns(2)


with col1:

    pesquisa = st.text_input(
        "Pesquisar jogo",
        placeholder="Digite o nome de um jogo..."
    )


with col2:

    quantidade_linhas = st.slider(
        "Quantidade de resultados",
        min_value=5,
        max_value=100,
        value=20
    )


df_tabela = df_filtrado.copy()


if pesquisa:

    df_tabela = df_tabela[
        df_tabela["Name"]
        .str.contains(
            pesquisa,
            case=False,
            na=False,
            regex=False
        )
    ]


df_tabela = (
    df_tabela
    .sort_values(
        "Global_Sales",
        ascending=False
    )
    .head(quantidade_linhas)
)


colunas_tabela = [
    "Rank",
    "Name",
    "Platform",
    "Year",
    "Genre",
    "Publisher",
    "NA_Sales",
    "EU_Sales",
    "JP_Sales",
    "Other_Sales",
    "Global_Sales"
]


if not df_tabela.empty:

    st.dataframe(
        df_tabela[colunas_tabela],
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "Nenhum jogo encontrado."
    )


# ============================================================
# DOWNLOAD
# ============================================================

csv_download = (
    df_tabela
    .to_csv(index=False)
    .encode("utf-8")
)


st.download_button(
    label="📥 Baixar dados filtrados",
    data=csv_download,
    file_name="games_filtrados.csv",
    mime="text/csv"
)

# ============================================================
# INTERPRETAÇÃO INTELIGENTE DA PERGUNTA
# ============================================================

def remover_acentos(texto):
    texto = unicodedata.normalize(
        "NFD",
        str(texto)
    )

    return "".join(
        caractere
        for caractere in texto
        if unicodedata.category(caractere) != "Mn"
    )


def normalizar_pergunta(pergunta):
    """
    Corrige pequenos erros de digitação, ausência de acentos
    e palavras incompletas antes de enviar a pergunta para a IA.
    """

    texto_original = str(pergunta).strip()

    if not texto_original:
        return texto_original

    # --------------------------------------------------------
    # Padronização
    # --------------------------------------------------------

    texto = texto_original.lower()

    texto = re.sub(
        r"\s+",
        " ",
        texto
    ).strip()

    # --------------------------------------------------------
    # Correções diretas comuns
    # --------------------------------------------------------

    correcoes = {
        "qua": "qual",
        "q": "qual",
        "qul": "qual",
        "jog": "jogo",
        "jgs": "jogos",
        "jogosss": "jogos",
        "gen": "genero",
        "gener": "genero",
        "genero": "genero",
        "plataform": "plataforma",
        "plat": "plataforma",
        "publish": "publisher",
        "pubisher": "publisher",
        "publiher": "publisher",
        "vend": "venda",
        "vende": "vendeu",
        "mas": "mais",
        "maiorr": "maior",
        "melho": "melhor",
        "anoo": "ano",
        "regiao": "regiao",
    }

    palavras = texto.split()

    palavras_corrigidas = []

    for palavra in palavras:

        palavra_sem_acentos = remover_acentos(
            palavra
        )

        if palavra_sem_acentos in correcoes:

            palavra = correcoes[
                palavra_sem_acentos
            ]

        palavras_corrigidas.append(
            palavra
        )

    texto = " ".join(
        palavras_corrigidas
    )

    # --------------------------------------------------------
    # Vocabulário conhecido do projeto
    # --------------------------------------------------------

    vocabulario = [
        "qual",
        "jogo",
        "jogos",
        "vendeu",
        "vendas",
        "mais",
        "menos",
        "melhor",
        "maior",
        "plataforma",
        "plataformas",
        "genero",
        "generos",
        "publisher",
        "publishers",
        "regiao",
        "regioes",
        "ano",
        "anos",
        "media",
        "global",
        "total",
        "quantidade",
        "converta",
        "converter",
        "dolar",
        "dolares",
        "real",
        "reais",
        "euro",
        "euros"
    ]

    palavras = texto.split()

    resultado = []

    for palavra in palavras:

        if len(palavra) < 3:
            resultado.append(palavra)
            continue

        correspondencias = get_close_matches(
            palavra,
            vocabulario,
            n=1,
            cutoff=0.78
        )

        if correspondencias:

            palavra = correspondencias[0]

        resultado.append(
            palavra
        )

    texto = " ".join(resultado)

    # --------------------------------------------------------
    # Restaurar acentos apenas nas palavras conhecidas
    # --------------------------------------------------------

    acentos = {
        "genero": "gênero",
        "generos": "gêneros",
        "regiao": "região",
        "regioes": "regiões",
        "media": "média",
        "dolar": "dólar",
        "dolares": "dólares",
        "real": "real",
        "reais": "reais",
        "ano": "ano",
        "anos": "anos",
    }

    palavras = texto.split()

    palavras_finais = [
        acentos.get(
            palavra,
            palavra
        )
        for palavra in palavras
    ]

    texto = " ".join(
        palavras_finais
    )

    # --------------------------------------------------------
    # Primeira letra maiúscula
    # --------------------------------------------------------

    if texto:

        texto = (
            texto[0].upper()
            + texto[1:]
        )

    return texto

# ============================================================
# CONVERSÃO DE MOEDAS
# ============================================================

import json
from urllib.request import Request, urlopen
from urllib.parse import quote


def converter_moeda(valor, moeda_origem, moeda_destino):
    """
    Converte um valor usando a cotação mais recente
    disponibilizada pelo Frankfurter.
    """

    moeda_origem = moeda_origem.upper()
    moeda_destino = moeda_destino.upper()

    if moeda_origem == moeda_destino:
        return {
            "valor": valor,
            "taxa": 1.0,
            "data": None
        }

    url = (
        "https://api.frankfurter.dev/v2/rate/"
        f"{quote(moeda_origem.lower())}/"
        f"{quote(moeda_destino.lower())}"
    )

    try:

        requisicao = Request(
            url,
            headers={
                "User-Agent": "Game-Sales-Analytics/1.0"
            }
        )

        with urlopen(
            requisicao,
            timeout=10
        ) as resposta:

            dados = json.loads(
                resposta.read().decode("utf-8")
            )

        taxa = float(dados["rate"])

        valor_convertido = valor * taxa

        return {
            "valor": valor_convertido,
            "taxa": taxa,
            "data": dados.get("date")
        }

    except Exception as erro:

        return {
            "erro": str(erro)
        }


# ============================================================
# CHAT IA — QWEN 3:4B
# ============================================================

# ------------------------------------------------------------
# CABEÇALHO
# ------------------------------------------------------------

col_chat_titulo, col_chat_sugestoes = st.columns(
    [3, 1]
)


with col_chat_titulo:

    st.subheader(
        "🤖 Assistente de IA para Games"
    )

    st.caption(
        "Pergunte sobre jogos, plataformas, gêneros, "
        "publishers, vendas e moedas."
    )


with col_chat_sugestoes:

    with st.popover("💡 Perguntas sugeridas"):

        st.markdown(
            "**Experimente uma destas perguntas:**"
        )

        sugestoes = [
            "Qual plataforma vendeu mais?",
            "Qual jogo vendeu mais?",
            "Qual gênero vende mais?",
            "Qual publisher vendeu mais?",
            "Qual foi o melhor ano em vendas?",
            "Qual plataforma tem maior média por jogo?",
            "Qual gênero tem maior média de vendas?",
            "Qual região vendeu mais?",
            "Quais são os 10 jogos mais vendidos?",
            "Converta 100 dólares para reais?"
        ]

        for i, sugestao in enumerate(sugestoes):

            if st.button(
                sugestao,
                key=f"sugestao_chat_{i}",
                use_container_width=True
            ):

                st.session_state.pergunta_sugerida = sugestao

                st.rerun()


# ============================================================
# MEMÓRIA
# ============================================================

if "mensagens_game_ai" not in st.session_state:

    st.session_state.mensagens_game_ai = []


# ============================================================
# HISTÓRICO
# ============================================================

for mensagem in st.session_state.mensagens_game_ai:

    with st.chat_message(
        mensagem["role"]
    ):

        st.write(
            mensagem["content"]
        )


# ============================================================
# ENTRADA
# ============================================================

pergunta_digitada = st.chat_input(
    "Ex.: Qual plataforma vendeu mais?"
)

pergunta_sugerida = st.session_state.pop(
    "pergunta_sugerida",
    None
)

pergunta = (
    pergunta_digitada
    if pergunta_digitada
    else pergunta_sugerida
)


# ============================================================
# PROCESSAMENTO
# ============================================================

if pergunta:

    # --------------------------------------------------------
    # Mostrar pergunta original
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.write(pergunta)


    # --------------------------------------------------------
    # Guardar pergunta original
    # --------------------------------------------------------

    st.session_state.mensagens_game_ai.append(
        {
            "role": "user",
            "content": pergunta
        }
    )


    # --------------------------------------------------------
    # Normalizar pergunta
    # --------------------------------------------------------

    pergunta_interpretada = normalizar_pergunta(
        pergunta
    )


    # --------------------------------------------------------
    # Mostrar interpretação quando houver alteração
    # --------------------------------------------------------

    if pergunta_interpretada.strip().lower() != pergunta.strip().lower():

        st.caption(
            f"🧠 Entendi como: {pergunta_interpretada}"
        )


    # ========================================================
    # CONTEXTO REAL DO PYTHON
    # ========================================================

    contexto_dados = gerar_contexto_ia(
        df_filtrado,
        pergunta_interpretada
    )


    # ========================================================
    # PROMPT DO SISTEMA
    # ========================================================

    sistema = """
Você é o assistente de IA de um projeto de análise de dados
sobre videogames.

Seu assunto principal é videogames e os dados do projeto.

Você pode responder sobre:

- jogos
- plataformas
- consoles
- gêneros
- publishers
- vendas
- regiões
- anos
- análise de dados de games
- desenvolvimento de jogos
- história dos videogames

REGRAS:

1. Quando a pergunta estiver relacionada ao dataset,
use exclusivamente os resultados calculados pelo Python.

2. Nunca invente números.

3. Nunca invente resultados.

4. Os valores de vendas estão em milhões de unidades.

5. Se os dados fornecidos não forem suficientes,
explique claramente.

6. O usuário pode escrever com erros de digitação,
sem acentos ou palavras incompletas.
Use a pergunta interpretada como referência.

7. Responda sempre em português.

8. Seja objetivo.

9. Não diga que você calculou os números.
Os cálculos foram realizados pelo Python.

10. Se a pergunta não tiver relação com videogames,
games ou análise dos dados do projeto, responda:

"Sou uma IA especializada em videogames e neste projeto só posso
responder perguntas relacionadas ao universo dos games."

11. Não transforme Global_Sales em dinheiro.
As vendas do dataset representam milhões de unidades vendidas.

12. Se uma informação não estiver nos dados fornecidos,
não invente.
"""


    # ========================================================
    # MENSAGENS
    # ========================================================

    mensagens = [
        {
            "role": "system",
            "content": sistema
        },
        {
            "role": "system",
            "content": (
                "RESULTADOS REAIS CALCULADOS PELO PYTHON:\n\n"
                + contexto_dados
            )
        }
    ]


    # --------------------------------------------------------
    # Histórico recente
    # --------------------------------------------------------

    historico = (
        st.session_state
        .mensagens_game_ai[-10:]
    )

    mensagens.extend(
        historico
    )


    # ========================================================
    # CHAMADA DO OLLAMA
    # ========================================================

    try:

        with st.chat_message("assistant"):

            with st.spinner("🤖 Analisando os dados..."):

                resposta = ollama.chat(
                    model="qwen3:4b",
                    messages=mensagens
                )

                texto_resposta = (
                    resposta["message"]["content"]
                )

            st.write(
                texto_resposta
            )


        # ----------------------------------------------------
        # Guardar resposta
        # ----------------------------------------------------

        st.session_state.mensagens_game_ai.append(
            {
                "role": "assistant",
                "content": texto_resposta
            }
        )


    except Exception as erro:

        st.error(
            "❌ Erro ao processar a pergunta."
        )

        st.code(
            str(erro)
        )

        st.info(
            "O Ollama precisa estar funcionando com o modelo "
            "qwen3:4b."
        )