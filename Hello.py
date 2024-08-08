import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Carregar o dataset
df = pd.read_csv('PEDE_PASSOS_DATASET_FIAP.csv', delimiter=';')

# Funções de filtragem e limpeza de dados
def filter_columns(df, filters: list):
    selected_columns = [True] * len(df.columns)  # Inicializa todas as colunas como True
    for index, column in enumerate(df.columns):
        if any(filter in column for filter in filters): 
            selected_columns[index] = False
    return df[df.columns[selected_columns]]

def cleaning_dataset(df):
    _df = df.dropna(subset=df.columns.difference(['NOME']), how='all')  # Remove linhas com todas as colunas NaN, exceto a coluna NOME
    _df = _df[~_df.isna().all(axis=1)]  # Remove linhas com apenas NaN, se tiver algum dado na linha não remove
    return _df

# Aplicar a filtragem e limpeza para os dados de 2020, 2021 e 2022
df_2020 = filter_columns(df, ['2021', '2022'])
df_2020 = cleaning_dataset(df_2020)
df_2021 = filter_columns(df, ['2020', '2022'])
df_2021 = cleaning_dataset(df_2021)
df_2022 = filter_columns(df, ['2020', '2021'])
df_2022 = cleaning_dataset(df_2022)

# Contar a quantidade de alunos por ano
students_count = {
    '2020': len(df_2020),
    '2021': len(df_2021),
    '2022': len(df_2022)
}

# Converter para DataFrame para visualização
students_df = pd.DataFrame(list(students_count.items()), columns=['Year', 'Number of Students'])

# Função principal para a página inicial do Streamlit
def run():
    st.set_page_config(
        page_title="Passos Mágicos ✨",
        page_icon="📊",
    )

    st.write("# Bem-vindo ao Datathon Passos Mágicos! ✨")

    st.sidebar.success("Selecione uma demonstração acima. 🚀")

    st.markdown(
        """
        A ONG **Passos Mágicos** é dedicada a transformar vidas através da educação e do engajamento comunitário. 🌟
        Nosso objetivo é fornecer um ambiente acolhedor e estimulante para jovens de todas as idades, 
        promovendo o desenvolvimento acadêmico e pessoal. 📚💡

        **👈 Selecione uma demonstração na barra lateral** para ver alguns exemplos do que podemos fazer com dados!
        
        ### Quer saber mais?
        - Visite nosso [site](https://passosmagicos.org.br/)
        - Explore nossa [documentação](https://passosmagicos.org.br/impacto-e-transparencia/)
        - Faça uma pergunta em nossos [fóruns da comunidade](https://passosmagicos.org.br/contato/)
    """
    )

    st.write("## Quantidade de Alunos ao Longo do Tempo 📈")

    st.markdown("""
    Durante essa análise vamos não só avaliar a quantidade de alunos ao longo do tempo, como também identificar padrões nos indicadores de desenvolvimento, 
    desenvolver modelos preditivos e colocar eles em prática.
    """)
    
    # Plotar o gráfico de barras
    fig, ax = plt.subplots()
    ax.bar(students_df['Year'], students_df['Number of Students'], color='skyblue')
    ax.set_xlabel('Ano')
    ax.set_ylabel('Número de Alunos')
    ax.set_title('Número de Alunos por Ano')
    
    # Adicionar os valores no topo de cada barra
    for i, v in enumerate(students_df['Number of Students']):
        ax.text(i, v + 5, str(v), ha='center', va='bottom', fontweight='bold')
    
    st.pyplot(fig)

if __name__ == "__main__":
    run()
