import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt

# Configurar a página do Streamlit
st.set_page_config(
    page_title="Modelo Preditivo do Ponto de Virada 📈",
    page_icon="🤖",
)

def run():
    st.write("# Treinamento de Modelo para Prever Ponto de Virada 📈")

    st.sidebar.success("Selecione uma demonstração acima. 🚀")

    st.markdown(
        '''
        Como vimos anteriormente, valores de anos anteriores são totalmente influentes ao ponto de virada e as demais variáveis, nessa página iremos mostrar
        a análise de performance de um modelo treinado com esses dados. Já na página seguinte, colocaremos em prática esse modelo e faremos juntos, previsões.
        Prevendo assim, se com determinadas notas, um aluno atingirá ou não o ponto de virada. 🚀
        '''
    )

    # Carregar o dataset final merged
    df = pd.read_csv('Final_Merged_DataFrame.csv')

    # Remover nulos na coluna target
    df.dropna(subset=['PONTO_VIRADA_2022'], inplace=True)

    # Converter colunas relevantes para numérico, tratando valores de string
    label_encoders = {}
    for column in df.columns:
        # Remover nulos antes de codificar
        df[column].dropna(inplace=True)
        if df[column].dtype == 'object':
            label_encoders[column] = LabelEncoder()
            df[column] = label_encoders[column].fit_transform(df[column])
        else:
            df[column] = pd.to_numeric(df[column], errors='coerce')

    # Verificar se o target é binário
    unique_targets = df['PONTO_VIRADA_2022'].unique()
    if len(unique_targets) > 2:
        st.warning("O target 'PONTO_VIRADA_2022' contém mais de duas classes. Verifique os dados.")

    # Selecionar colunas de performance para treino
    performance_columns = [
        'PEDRA_2021', 'INDE_2021', 'IAA_2021', 'IEG_2021', 'IPS_2021', 
        'IDA_2021', 'IPP_2021', 'IPV_2021', 'IAN_2021'
    ]

    target = 'PONTO_VIRADA_2022'
    st.markdown("## Modelo de predição de Ponto de Virada")

    # Remover linhas com valores ausentes nas colunas selecionadas
    df_performance = df.dropna(subset=performance_columns + [target])

    X = df_performance[performance_columns]
    y = df_performance[target]

    # Dividir os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inicializar e treinar o modelo de Gradient Boosting
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Prever e avaliar o desempenho
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary', zero_division=1)
    recall = recall_score(y_test, y_pred, average='binary', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='binary', zero_division=1)

    # Exibir métricas de desempenho
    st.write(f"### Gradient Boosting")
    st.write(f"**Acurácia:** {accuracy:.4f}")
    st.write(f"**Precisão:** {precision:.4f}")
    st.write(f"**Revocação:** {recall:.4f}")
    st.write(f"**F1 Score:** {f1:.4f}")

    # Salvar o modelo treinado
    joblib.dump(model, 'gradient_boosting_model.pkl')

    st.markdown(
        '''
        Aqui temos performance de um modelo treinado para prever o Ponto de Virada dos alunos, 
        na próxima página iremos explorar esse modelo e colocar ele pra uso com dados inputados manualmente. 
        '''
    )

    st.markdown("## Modelo de Regressão para prever INDE_2022")

    # Remover nulos na coluna target
    df.dropna(subset=['INDE_2022'], inplace=True)

    target_regression = 'INDE_2022'

    # Remover linhas com valores ausentes nas colunas selecionadas
    df_performance_regression = df.dropna(subset=performance_columns + [target_regression])

    X_regression = df_performance_regression[performance_columns]
    y_regression = df_performance_regression[target_regression]

    # Dividir os dados em conjuntos de treino e teste
    X_train_regression, X_test_regression, y_train_regression, y_test_regression = train_test_split(X_regression, y_regression, test_size=0.2, random_state=42)

    # Inicializar e treinar o modelo de Gradient Boosting para regressão
    model_regression = GradientBoostingRegressor(random_state=42)
    model_regression.fit(X_train_regression, y_train_regression)

    # Prever e avaliar o desempenho
    y_pred_regression = model_regression.predict(X_test_regression)
    mae = mean_absolute_error(y_test_regression, y_pred_regression)
    r2 = r2_score(y_test_regression, y_pred_regression)
    wape = 1 - (mae / y_test_regression.mean())

    # Exibir métricas de desempenho
    st.write(f"### Gradient Boosting - Regressão")
    st.write(f"**MAE:** {mae:.4f}")
    st.write(f"**R² Score:** {r2:.4f}")
    st.write(f"**1-WAPE:** {wape:.4f}")

    # Salvar o modelo treinado
    joblib.dump(model_regression, 'gradient_boosting_model_regression.pkl')

    st.markdown(
        '''
        Aqui temos a performance de um modelo treinado para prever o INDE_2022 dos alunos. Assim como o modelo anterior, vamos por esse em prática
        na página seguinte.  
        '''
    )

if __name__ == "__main__":
    run()
