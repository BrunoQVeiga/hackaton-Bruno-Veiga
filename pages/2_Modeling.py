import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Configurar a página do Streamlit
st.set_page_config(
    page_title="Modelo Preditivo de IPV 📈",
    page_icon="🤖",
)

def run():
    st.write("# Treinamento de Modelo para Prever IPV, INDE, PEDRA e PONTO_VIRADA 📈")

    st.sidebar.success("Selecione uma demonstração acima. 🚀")

    st.markdown(
        '''
        Nesta página, vamos utilizar os dados de performance anteriores para treinar modelos que preveem os valores de IPV, INDE, PEDRA e PONTO_VIRADA de 2022.
        Podemos replicar e retreinar estes modelos para dados mais atuais, como usar dados de 2022 e 2023 para prever os valores de 2024. 🚀
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

    # Selecionar colunas de performance para treino
    performance_columns = ['INDE_2020', 'PEDRA_2020', 'IAA_2020', 'IEG_2020', 'IPS_2020', 'IDA_2020', 'IPP_2020', 'IPV_2020', 'IAN_2020', 
                           'PEDRA_2021', 'INDE_2021', 'IAA_2021', 'IEG_2021', 'IPS_2021', 'IDA_2021', 'IPP_2021', 'IPV_2021', 'IAN_2021']

    targets = ['IPV_2022', 'INDE_2022', 'PEDRA_2022', 'PONTO_VIRADA_2022']

    for target in targets:
        st.markdown(f"## Treinamento de Modelo para {target}")

        # Remover linhas com valores ausentes nas colunas selecionadas
        df_performance = df.dropna(subset=performance_columns + [target])

        X = df_performance[performance_columns]
        y = df_performance[target]

        # Dividir os dados em conjuntos de treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Treinar vários modelos e avaliar o desempenho
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "Support Vector Regressor": SVR()
        }

        best_model_name = None
        best_model = None
        best_r2 = -float('inf')
        best_mse = float('inf')

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write(f"### {name}")
            st.write(f"**Erro Quadrático Médio (MSE):** {mse}")
            st.write(f"**Coeficiente de Determinação (R²):** {r2}")
            if r2 > best_r2:
                best_r2 = r2
                best_mse = mse
                best_model_name = name
                best_model = model

        st.write(f"## Melhor Modelo para {target}: {best_model_name}")
        st.write(f"**Erro Quadrático Médio (MSE):** {best_mse}")
        st.write(f"**Coeficiente de Determinação (R²):** {best_r2}")

        st.markdown(
            '''
            Podemos observar que as performances anteriores são variáveis importantes para prever os valores dos alunos.
            Os modelos treinados podem ser utilizados para prever valores futuros com base em dados mais recentes.
            '''
        )

if __name__ == "__main__":
    run()
