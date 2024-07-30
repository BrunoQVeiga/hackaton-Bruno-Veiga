import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Configurar a página do Streamlit
st.set_page_config(
    page_title="Análise de Indicadores 📚",
    page_icon="✨",
)

# Carregar o dataset
df = pd.read_csv('Final_Merged_DataFrame.csv')

# Converter colunas relevantes para numérico
df['IPV_2020'] = pd.to_numeric(df['IPV_2020'], errors='coerce')
df['IPV_2021'] = pd.to_numeric(df['IPV_2021'], errors='coerce')
df['IPV_2022'] = pd.to_numeric(df['IPV_2022'], errors='coerce')
df['INDE_2020'] = pd.to_numeric(df['INDE_2020'], errors='coerce')
df['INDE_2021'] = pd.to_numeric(df['INDE_2021'], errors='coerce')
df['INDE_2022'] = pd.to_numeric(df['INDE_2022'], errors='coerce')

# Calcular a média do IPV por ano
ipv_mean = {
    '2020': df['IPV_2020'].mean(),
    '2021': df['IPV_2021'].mean(),
    '2022': df['IPV_2022'].mean()
}

# Calcular a média do INDE por ano
inde_mean = {
    '2020': df['INDE_2020'].mean(),
    '2021': df['INDE_2021'].mean(),
    '2022': df['INDE_2022'].mean()
}

# Contar a quantidade de "Sim" e "Não" para PONTO_VIRADA por ano
ponto_virada_count = {
    '2020': df['PONTO_VIRADA_2020'].value_counts(),
    '2021': df['PONTO_VIRADA_2021'].value_counts(),
    '2022': df['PONTO_VIRADA_2022'].value_counts()
}

# Contar a quantidade de cada pedra por ano
pedra_count_2020 = df['PEDRA_2020'].value_counts().reset_index().rename(columns={'PEDRA_2020': 'Pedra', 'count': '2020'})
pedra_count_2021 = df['PEDRA_2021'].value_counts().reset_index().rename(columns={'PEDRA_2021': 'Pedra', 'count': '2021'})
pedra_count_2022 = df['PEDRA_2022'].value_counts().reset_index().rename(columns={'PEDRA_2022': 'Pedra', 'count': '2022'})
# Mesclar os DataFrames de pedras em um único DataFrame
pedra_counts = pd.merge(pedra_count_2020, pedra_count_2021, on='Pedra', how='outer')
pedra_counts = pd.merge(pedra_counts, pedra_count_2022, on='Pedra', how='outer').fillna(0)
# Contar a frequência de cada valor de "Pedra" em todos os anos combinados
pedra_counts['Total'] = pedra_counts[['2020', '2021', '2022']].sum(axis=1)
# Filtrar para manter apenas "Pedras" que aparecem mais de uma vez no total
pedra_counts = pedra_counts[pedra_counts['Total'] > 5]
# Remover a coluna "Total" após a filtragem
pedra_counts = pedra_counts.drop(columns=['Total'])

# Converter para DataFrame para visualização
ipv_df = pd.DataFrame(list(ipv_mean.items()), columns=['Year', 'IPV_Mean'])
inde_df = pd.DataFrame(list(inde_mean.items()), columns=['Year', 'INDE_Mean'])

ponto_virada_df = pd.DataFrame.from_dict(ponto_virada_count, orient='index').transpose().fillna(0)
# Contar a frequência de cada valor de "Pedra" em todos os anos combinados
ponto_virada_df['Total'] = ponto_virada_df[['2020', '2021', '2022']].sum(axis=1)
# Filtrar para manter apenas "Pedras" que aparecem mais de uma vez no total
ponto_virada_df = ponto_virada_df[ponto_virada_df['Total'] > 5]
# Remover a coluna "Total" após a filtragem
ponto_virada_df = ponto_virada_df.drop(columns=['Total'])

# Função principal para a página do Streamlit
def run():
    st.write("# Entendendo o Ponto de Virada (IPV) ✨")

    st.sidebar.success("Selecione uma demonstração acima. 🚀")

    st.markdown(
        '''
        O conceito de **Ponto de Virada (IPV)** é uma métrica crucial utilizada pela ONG Passos Mágicos. 🌟
        Este indicador revela a mudança de mindset do aluno, quando ele passa a acreditar na importância do estudo em sua vida. 📈
        
        ### O que é o IPV?
        O IPV, ou Índice de Ponto de Virada, é uma variável que quantifica o momento em que o aluno começa a valorizar o aprendizado,
        percebendo o impacto positivo que a educação pode ter em seu futuro. É uma medida de transformação e engajamento. 💡
        
        ### Média do IPV por Ano
        '''
    )

    # Plotar o gráfico da média do IPV
    fig, ax = plt.subplots()
    ax.plot(ipv_df['Year'], ipv_df['IPV_Mean'], marker='o', linestyle='-', color='g')
    ax.set_xlabel('Ano')
    ax.set_ylabel('Média do IPV')
    ax.set_title('Média do IPV por Ano')
    st.pyplot(fig)

    st.markdown("### Quantidade de alunos que atingiram o Ponto de Virada por Ano")

    # Plotar o gráfico de PONTO_VIRADA
    fig, ax = plt.subplots()
    ponto_virada_df.plot(kind='bar', ax=ax)
    ax.set_xlabel('Ano')
    ax.set_ylabel('Quantidade')
    ax.set_title('Quantidade de alunos que atingiram o Ponto de Virada por Ano')
    st.pyplot(fig)

    st.write("# Entendendo o Indíce de Desenvolvimento Educacional (INDE) ✨")

    st.sidebar.success("Selecione uma demonstração acima. 🚀")

    st.markdown(
        '''
        O Índice de Desenvolvimento Educacional (INDE) é uma métrica usada para avaliar o desempenho educacional dos alunos nos programas da Associação Passos Mágicos, calculado a partir de vários indicadores como desempenho acadêmico, engajamento, autoavaliação, entre outros.

        Os alunos são classificados em quatro categorias, chamadas de "pedras", com base nos valores do INDE:

        Quartzo: 2,405 a 5,506
        Ágata: 5,506 a 6,868
        Ametista: 6,868 a 8,230
        Topázio: 8,230 a 9,294
        Essas categorias ajudam a identificar o nível de desenvolvimento dos alunos, permitindo intervenções educacionais mais eficazes.
        '''
    )

    st.markdown("### Média do INDE por Ano")

    # Plotar o gráfico da média do INDE
    fig, ax = plt.subplots()
    ax.plot(inde_df['Year'], inde_df['INDE_Mean'], marker='o', linestyle='-', color='b')
    ax.set_xlabel('Ano')
    ax.set_ylabel('Média do INDE')
    ax.set_title('Média do INDE por Ano')
    st.pyplot(fig)

    st.markdown("### Quantidade de alunos em cada Pedra por Ano")

    # Plotar o gráfico da quantidade de cada pedra
    fig, ax = plt.subplots()
    pedra_counts.plot(kind='bar', x='Pedra', ax=ax, alpha=0.75)
    ax.set_title('Quantidade de alunos em cada Pedra por Ano')
    ax.set_xlabel('Pedra')
    ax.set_ylabel('Quantidade')
    st.pyplot(fig)

if __name__ == "__main__":
    run()