import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Configurar a página do Streamlit
st.set_page_config(
    page_title="Correlação",
    page_icon="📈",
)

# Função principal para a terceira página do Streamlit
def run():
    st.write("# Análise de Fatores Históricos que Influenciam o IPV 📊")

    st.sidebar.success("Selecione uma demonstração acima. 🚀")

    st.markdown(
        '''
        Nesta página, vamos explorar os fatores históricos que influenciam o **Ponto de Virada (IPV)** dos alunos. 📈
        Vamos visualizar a correlação entre diversas variáveis ao longo dos anos e como elas impactam o IPV. 💡
        
        ### Análise de Correlação
        A matriz de correlação a seguir mostra a relação entre diferentes variáveis e o IPV:
        '''
    )

    # Carregar o dataset final merged
    df = pd.read_csv('PEDE_PASSOS_DATASET_FIAP.csv', delimiter=';')

    # Converter colunas relevantes para numérico, tratando valores de string
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            label_encoders[column] = LabelEncoder()
            df[column] = label_encoders[column].fit_transform(df[column])
        else:
            df[column] = pd.to_numeric(df[column], errors='coerce')

    # Listas de colunas para análise
    performance_columns = ['INDE_2020', 'PEDRA_2020', 'IAA_2020', 'IEG_2020', 'IPS_2020', 'IDA_2020', 'IPP_2020', 'IPV_2020', 'IAN_2020', 
                           'PEDRA_2021', 'INDE_2021', 'IAA_2021', 'IEG_2021', 'IPS_2021', 'IDA_2021', 'IPP_2021', 'IPV_2021', 'IAN_2021']

    # Verificar a presença de IPV_2022 e outras colunas nas listas
    df['IPV_2022'] = pd.to_numeric(df['IPV_2022'], errors='coerce')
    df['INDE_2022'] = pd.to_numeric(df['INDE_2022'], errors='coerce')

    # Calcular correlações de performance anterior
    performance_corr_matrix = df[performance_columns + ['IPV_2022']].corr()
    performance_corr_with_ipv_2022 = performance_corr_matrix['IPV_2022'].drop('IPV_2022', errors='ignore').abs()
    top3_performance_corr = performance_corr_with_ipv_2022.sort_values(ascending=False).head(3)

    # Plotar a matriz de correlação de performance anterior
    st.markdown("### Matriz de Correlação de Performance Anterior")
    plt.figure(figsize=(15, 10))
    sns.heatmap(performance_corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Matriz de Correlação de Performance Anterior com IPV_2022')
    st.pyplot(plt)

    st.write("Top 3 correlações de performance anterior com IPV_2022:", top3_performance_corr)

    # Calcular correlações de INDE_2022
    inde_corr_matrix = df[performance_columns + ['INDE_2022']].corr()
    inde_corr_with_inde_2022 = inde_corr_matrix['INDE_2022'].drop('INDE_2022', errors='ignore').abs()
    top3_inde_corr = inde_corr_with_inde_2022.sort_values(ascending=False).head(3)

    # Calcular correlações de PEDRA_2022
    pedra_corr_matrix = df[performance_columns + ['PEDRA_2022']].corr()
    pedra_corr_with_pedra_2022 = pedra_corr_matrix['PEDRA_2022'].drop('PEDRA_2022', errors='ignore').abs()
    top3_pedra_corr = pedra_corr_with_pedra_2022.sort_values(ascending=False).head(3)

    # Calcular correlações de PONTO_VIRADA_2022
    ponto_virada_corr_matrix = df[performance_columns + ['PONTO_VIRADA_2022']].corr()
    ponto_virada_corr_with_ponto_virada_2022 = ponto_virada_corr_matrix['PONTO_VIRADA_2022'].drop('PONTO_VIRADA_2022', errors='ignore').abs()
    top3_ponto_virada_corr = ponto_virada_corr_with_ponto_virada_2022.sort_values(ascending=False).head(3)

    # Plotar a matriz de correlação de INDE_2022
    st.markdown("### Matriz de Correlação de INDE_2022")
    plt.figure(figsize=(15, 10))
    sns.heatmap(inde_corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Matriz de Correlação de INDE_2022')
    st.pyplot(plt)

    st.write("Top 3 correlações de INDE_2022:", top3_inde_corr)

    # Plotar a matriz de correlação de PEDRA_2022
    st.markdown("### Matriz de Correlação de PEDRA_2022")
    plt.figure(figsize=(15, 10))
    sns.heatmap(pedra_corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Matriz de Correlação de PEDRA_2022')
    st.pyplot(plt)

    st.write("Top 3 correlações de PEDRA_2022:", top3_pedra_corr)

    # Plotar a matriz de correlação de PONTO_VIRADA_2022
    st.markdown("### Matriz de Correlação de PONTO_VIRADA_2022")
    plt.figure(figsize=(15, 10))
    sns.heatmap(ponto_virada_corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Matriz de Correlação de PONTO_VIRADA_2022')
    st.pyplot(plt)

    st.write("Top 3 correlações de PONTO_VIRADA_2022:", top3_ponto_virada_corr)

    # Adicionar uma seção de conclusão
    st.markdown("## Conclusão")

    st.markdown(
        '''
        Com base na análise de correlação realizada, podemos observar que algumas avaliações passadas estão fortemente correlacionadas com as avaliações futuras. 
        Em particular, os indíces de desempenho dos anos anteriores, mostram uma correlação significativa com os resultados de 2022.
        
        ### Implicações para Modelagem Preditiva

        - **Potencial de Predição**: As altas correlações observadas indicam que essas variáveis históricas podem ser excelentes preditores para modelos de previsão de desempenho futuro dos alunos.
        - **Modelos de Machine Learning**: Podemos utilizar essas variáveis como features em modelos de aprendizado de máquina, como regressões, árvores de decisão e redes neurais, para prever o desempenho futuro.
        - **Intervenções Educacionais**: As correlações podem também guiar intervenções educacionais, ajudando a identificar áreas onde os alunos podem precisar de mais apoio.

        Em suma, compreender como as avaliações passadas influenciam as futuras nos fornece uma base sólida para desenvolver estratégias de ensino mais eficazes e modelos preditivos robustos.
        '''
    )

if __name__ == "__main__":
    run()

