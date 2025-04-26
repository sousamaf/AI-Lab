import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Análise do Formulário de Projetos de IA", layout="wide")

st.title("📊 Análise do Formulário de Escolha de Projeto – Inteligência Artificial")
st.markdown("Este painel permite visualizar e explorar as respostas coletadas dos alunos para a escolha do projeto da disciplina de IA.")

uploaded_file = st.file_uploader("📁 Envie o arquivo .CSV exportado do Google Forms", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Renomear colunas para facilitar leitura
    df.columns = [col.strip() for col in df.columns]
    col_map = {
        'Você tem familiaridade com Python?': 'Familiaridade Python',
        'Você se sente confortável trabalhando com dados reais e sujos (incompletos ou não padronizados)?': 'Conforto com Dados Sujos',
        'Selecione a área de projeto que mais lhe interessa:': 'Área de Interesse',
        'Justifique sua escolha (interesse pessoal, familiaridade, curiosidade etc.)': 'Justificativa',
        'Tem interesse em usar o Supercomputador Santos Dumont neste projeto?': 'Interesse em HPC'
    }
    df.rename(columns=col_map, inplace=True)

    st.subheader("Distribuição por Área de Interesse")
    fig1 = px.histogram(df, x="Área de Interesse", color="Área de Interesse", title="Preferências de Projeto",
                        category_orders={"Área de Interesse": sorted(df["Área de Interesse"].unique())})
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Familiaridade com Python")
    fig2 = px.pie(df, names="Familiaridade Python", title="Nível de Familiaridade com Python")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Interesse no Uso do Supercomputador Santos Dumont")
    fig3 = px.bar(df, x="Interesse em HPC", color="Interesse em HPC", title="Distribuição de Interesse em HPC")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Conforto com Dados Sujos")
    fig4 = px.pie(df, names="Conforto com Dados Sujos", title="Conforto ao Trabalhar com Dados Não Tratados")
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Visualizar Respostas Textuais (Justificativas)")
    st.dataframe(df[['Nome completo', 'Área de Interesse', 'Justificativa']].dropna())
else:
    st.info("Envie o arquivo de respostas do formulário exportado como CSV para iniciar a análise.")