import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Configurar a página do Streamlit
st.set_page_config(
    page_title="Análise de Indicadores 📚",
    page_icon="✨",
)

# Carregar o dataset
df = pd.read_csv('PEDE_PASSOS_DATASET_FIAP.csv', delimiter=';')

# Parâmetros que vamos analisar
parameters = ['IPV', 'INDE', 'IAA', 'IEG', 'IPS', 'IDA', 'IPP', 'IAN']

# Converter colunas relevantes para numérico
for param in parameters:
    for year in ['2020', '2021', '2022']:
        col_name = f'{param}_{year}'
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')

# Função para calcular a média de um parâmetro por ano
def calculate_mean(param):
    return {
        '2020': df[f'{param}_2020'].mean(),
        '2021': df[f'{param}_2021'].mean(),
        '2022': df[f'{param}_2022'].mean()
    }

# Calcular as médias para cada parâmetro
means = {param: calculate_mean(param) for param in parameters}

# Função para criar DataFrame para um parâmetro
def create_dataframe(param):
    return pd.DataFrame(list(means[param].items()), columns=['Year', f'{param}_Mean'])

# Criar DataFrames para cada parâmetro
dataframes = {param: create_dataframe(param) for param in parameters}

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

# Contar a quantidade de "Sim" e "Não" para PONTO_VIRADA por ano
ponto_virada_count = {
    '2020': df['PONTO_VIRADA_2020'].value_counts(),
    '2021': df['PONTO_VIRADA_2021'].value_counts(),
    '2022': df['PONTO_VIRADA_2022'].value_counts()
}

# Transformar em df para plot futuro
ponto_virada_df = pd.DataFrame.from_dict(ponto_virada_count, orient='index').transpose().fillna(0)
# Contar a frequência de cada valor de "Pedra" em todos os anos combinados
ponto_virada_df['Total'] = ponto_virada_df[['2020', '2021', '2022']].sum(axis=1)
# Filtrar para manter apenas "Pedras" que aparecem mais de uma vez no total
ponto_virada_df = ponto_virada_df[ponto_virada_df['Total'] > 5]
# Remover a coluna "Total" após a filtragem
ponto_virada_df = ponto_virada_df.drop(columns=['Total'])

# Descrições dos indicadores
descriptions = {
    'IPV': '''
        O conceito de **Ponto de Virada (IPV)** é uma métrica crucial utilizada pela ONG Passos Mágicos. 🌟
        Este indicador revela a mudança de mindset do aluno, quando ele passa a acreditar na importância do estudo em sua vida. 📈
        ''',
    'INDE': '''
        O Índice de Desenvolvimento Educacional (INDE) é uma métrica usada para avaliar o desempenho educacional dos alunos nos programas da Associação Passos Mágicos, calculado a partir de vários indicadores como desempenho acadêmico, engajamento, autoavaliação, entre outros.

        Os alunos são classificados em quatro categorias, chamadas de "pedras", com base nos valores do INDE:

        - **Quartzo**: 2,405 a 5,506
        - **Ágata**: 5,506 a 6,868
        - **Ametista**: 6,868 a 8,230
        - **Topázio**: 8,230 a 9,294

        Essas categorias ajudam a identificar o nível de desenvolvimento dos alunos, permitindo intervenções educacionais mais eficazes.
    ''',
    'IAA': '''
        O **Índice de Autoavaliação (IAA)** mede a percepção que o aluno tem de seu próprio desempenho e progresso nos estudos. 
        Este indicador é fundamental para compreender a autoconfiança e a motivação dos alunos em suas jornadas educacionais. 📚✨
    ''',
    'IEG': '''
        O **Índice de Engajamento (IEG)** reflete o nível de envolvimento e participação do aluno nas atividades escolares e extracurriculares. 
        Engajamento alto é indicativo de uma atitude positiva em relação à aprendizagem e à comunidade escolar. 🌟📈
    ''',
    'IPS': '''
        O **Índice Psicossocial (IPS)** avalia o bem-estar emocional e social dos alunos, considerando fatores como relacionamento com colegas, 
        adaptação ao ambiente escolar e suporte emocional. Este indicador ajuda a identificar necessidades de apoio psicológico. 💪❤️
    ''',
    'IDA': '''
        O **Índice de Desempenho Acadêmico (IDA)** fornece uma medida do desempenho acadêmico dos alunos com base em suas notas e avaliações. 
        Este indicador é crucial para avaliar a eficácia dos métodos de ensino e o progresso dos alunos nas disciplinas escolares. 🎓📊
    ''',
    'IPP': '''
        O **Índice Psicopedagógico (IPP)** integra aspectos psicopedagógicos, combinando desempenho acadêmico com fatores psicológicos que influenciam 
        a aprendizagem. É usado para entender melhor as dificuldades de aprendizagem e personalizar estratégias educacionais. 📘🧠
    ''',
    'IAN': '''
        O **Índice de Adequação de Nível (IAN)** verifica se o aluno está no nível de aprendizado adequado para sua faixa etária e ano escolar. 
        Ele identifica possíveis lacunas no conhecimento e áreas onde o aluno precisa de suporte adicional. 📈🔍
    '''
}

# Função para plotar gráficos
def plot_parameter(param, color):
    df = dataframes[param]
    fig, ax = plt.subplots()
    ax.plot(df['Year'], df[f'{param}_Mean'], marker='o', linestyle='-', color=color)
    ax.set_xlabel('Ano')
    ax.set_ylabel(f'Média do {param}')
    ax.set_title(f'Média do {param} por Ano')
    st.pyplot(fig)

# Função principal para a página do Streamlit
def run():
    st.write("# Análise de Indicadores Educacionais ✨")

    st.sidebar.success("Selecione uma demonstração acima. 🚀")

    for param in parameters:
        st.write(f"## {param}")
        st.markdown(descriptions[param])
        plot_parameter(param, 'b' if param in ['INDE', 'IAN'] else 'g')

    st.markdown("### Quantidade de alunos em cada Pedra por Ano")

    # Plotar o gráfico da quantidade de cada pedra
    fig, ax = plt.subplots()
    pedra_counts.plot(kind='bar', x='Pedra', ax=ax, alpha=0.75)
    ax.set_title('Quantidade de alunos em cada Pedra por Ano')
    ax.set_xlabel('Pedra')
    ax.set_ylabel('Quantidade')
    st.pyplot(fig)

    # Quantidade de Ponto de Virada por ano
    st.markdown("### Quantidade de alunos que atingiram o Ponto de Virada por Ano")

    # Plotar o gráfico de PONTO_VIRADA
    fig, ax = plt.subplots()
    ponto_virada_df.plot(kind='bar', ax=ax)
    ax.set_xlabel('Resposta')
    ax.set_ylabel('Quantidade')
    ax.set_title('Quantidade de alunos que atingiram o Ponto de Virada por Ano')
    st.pyplot(fig)

if __name__ == "__main__":
    run()

