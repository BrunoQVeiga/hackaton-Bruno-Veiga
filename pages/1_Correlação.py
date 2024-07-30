import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Configurar a página do Streamlit
st.set_page_config(
    page_title="Fatores Históricos do IPV 📊",
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
    df = pd.read_csv('Final_Merged_DataFrame.csv')

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

    personal_columns = ['DataNascimento', 'INSTITUICAO_ENSINO_ALUNO_2020', 'INSTITUICAO_ENSINO_ALUNO_2021', 'Sexo', 'EstadoCivil', 
                        'IdPai', 'IdMae', 'IdResponsavel', 'IdTipoResponsavel', 'Naturalidade', 'Nacionalidade', 'CorRaca', 
                        'EnsinoMedio_AnoConclusao', 'IdTipoResponsavelPai', 'IdTipoResponsavelMae', 'Contagem_F', 'Contagem_J', 'Contagem_P']

    # Verificar a presença de IPV_2022 e outras colunas nas listas
    df['IPV_2022'] = pd.to_numeric(df['IPV_2022'], errors='coerce')
    df['INDE_2022'] = pd.to_numeric(df['INDE_2022'], errors='coerce')

    # Calcular correlações de performance anterior
    performance_corr_matrix = df[performance_columns + ['IPV_2022']].corr()
    performance_corr_with_ipv_2022 = performance_corr_matrix['IPV_2022'].drop('IPV_2022', errors='ignore').abs()
    top3_performance_corr = performance_corr_with_ipv_2022.sort_values(ascending=False).head(3)

    # Calcular correlações de fatores pessoais
    personal_corr_matrix = df[personal_columns + ['IPV_2022']].corr()
    personal_corr_with_ipv_2022 = personal_corr_matrix['IPV_2022'].drop('IPV_2022', errors='ignore').abs()
    top3_personal_corr = personal_corr_with_ipv_2022.sort_values(ascending=False).head(3)

    # Plotar a matriz de correlação de performance anterior
    st.markdown("### Matriz de Correlação de Performance Anterior")
    plt.figure(figsize=(15, 10))
    sns.heatmap(performance_corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Matriz de Correlação de Performance Anterior com IPV_2022')
    st.pyplot(plt)

    st.write("Top 3 correlações de performance anterior com IPV_2022:", top3_performance_corr)

    # Plotar a matriz de correlação de fatores pessoais
    st.markdown("### Matriz de Correlação de Fatores Pessoais")
    plt.figure(figsize=(15, 10))
    sns.heatmap(personal_corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Matriz de Correlação de Fatores Pessoais com IPV_2022')
    st.pyplot(plt)

    st.write("Top 3 correlações de fatores pessoais com IPV_2022:", top3_personal_corr)

    # Calcular correlações de INDE_2022
    inde_corr_matrix = df[performance_columns + ['INDE_2022']].corr()
    inde_corr_with_inde_2022 = inde_corr_matrix['INDE_2022'].drop('INDE_2022', errors='ignore').abs()
    top3_inde_corr = inde_corr_with_inde_2022.sort_values(ascending=False).head(3)

    # Calcular correlações de fatores pessoais para INDE_2022
    personal_corr_matrix_inde = df[personal_columns + ['INDE_2022']].corr()
    personal_corr_with_inde_2022 = personal_corr_matrix_inde['INDE_2022'].drop('INDE_2022', errors='ignore').abs()
    top3_personal_corr_inde = personal_corr_with_inde_2022.sort_values(ascending=False).head(3)

    # Calcular correlações de PEDRA_2022
    pedra_corr_matrix = df[performance_columns + ['PEDRA_2022']].corr()
    pedra_corr_with_pedra_2022 = pedra_corr_matrix['PEDRA_2022'].drop('PEDRA_2022', errors='ignore').abs()
    top3_pedra_corr = pedra_corr_with_pedra_2022.sort_values(ascending=False).head(3)

    # Calcular correlações de fatores pessoais para PEDRA_2022
    personal_corr_matrix_pedra = df[personal_columns + ['PEDRA_2022']].corr()
    personal_corr_with_pedra_2022 = personal_corr_matrix_pedra['PEDRA_2022'].drop('PEDRA_2022', errors='ignore').abs()
    top3_personal_corr_pedra = personal_corr_with_pedra_2022.sort_values(ascending=False).head(3)

    # Calcular correlações de PONTO_VIRADA_2022
    ponto_virada_corr_matrix = df[performance_columns + ['PONTO_VIRADA_2022']].corr()
    ponto_virada_corr_with_ponto_virada_2022 = ponto_virada_corr_matrix['PONTO_VIRADA_2022'].drop('PONTO_VIRADA_2022', errors='ignore').abs()
    top3_ponto_virada_corr = ponto_virada_corr_with_ponto_virada_2022.sort_values(ascending=False).head(3)

    # Calcular correlações de fatores pessoais para PONTO_VIRADA_2022
    personal_corr_matrix_ponto_virada = df[personal_columns + ['PONTO_VIRADA_2022']].corr()
    personal_corr_with_ponto_virada_2022 = personal_corr_matrix_ponto_virada['PONTO_VIRADA_2022'].drop('PONTO_VIRADA_2022', errors='ignore').abs()
    top3_personal_corr_ponto_virada = personal_corr_with_ponto_virada_2022.sort_values(ascending=False).head(3)

    # Plotar a matriz de correlação de INDE_2022
    st.markdown("### Matriz de Correlação de INDE_2022")
    plt.figure(figsize=(15, 10))
    sns.heatmap(inde_corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Matriz de Correlação de INDE_2022')
    st.pyplot(plt)

    st.write("Top 3 correlações de INDE_2022:", top3_inde_corr)

    st.markdown("### Matriz de Correlação de Fatores Pessoais para INDE_2022")
    plt.figure(figsize=(15, 10))
    sns.heatmap(personal_corr_matrix_inde, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Matriz de Correlação de Fatores Pessoais com INDE_2022')
    st.pyplot(plt)

    st.write("Top 3 correlações de fatores pessoais com INDE_2022:", top3_personal_corr_inde)

        # Plotar a matriz de correlação de PEDRA_2022
    st.markdown("### Matriz de Correlação de PEDRA_2022")
    plt.figure(figsize=(15, 10))
    sns.heatmap(pedra_corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Matriz de Correlação de PEDRA_2022')
    st.pyplot(plt)

    st.write("Top 3 correlações de PEDRA_2022:", top3_pedra_corr)

    st.markdown("### Matriz de Correlação de Fatores Pessoais para PEDRA_2022")
    plt.figure(figsize=(15, 10))
    sns.heatmap(personal_corr_matrix_pedra, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Matriz de Correlação de Fatores Pessoais com PEDRA_2022')
    st.pyplot(plt)

    st.write("Top 3 correlações de fatores pessoais com PEDRA_2022:", top3_personal_corr_pedra)

    # Plotar a matriz de correlação de PONTO_VIRADA_2022
    st.markdown("### Matriz de Correlação de PONTO_VIRADA_2022")
    plt.figure(figsize=(15, 10))
    sns.heatmap(ponto_virada_corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Matriz de Correlação de PONTO_VIRADA_2022')
    st.pyplot(plt)

    st.write("Top 3 correlações de PONTO_VIRADA_2022:", top3_ponto_virada_corr)

    st.markdown("### Matriz de Correlação de Fatores Pessoais para PONTO_VIRADA_2022")
    plt.figure(figsize=(15, 10))
    sns.heatmap(personal_corr_matrix_ponto_virada, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Matriz de Correlação de Fatores Pessoais com PONTO_VIRADA_2022')
    st.pyplot(plt)

    st.write("Top 3 correlações de fatores pessoais com PONTO_VIRADA_2022:", top3_personal_corr_ponto_virada)

if __name__ == "__main__":
    run()
