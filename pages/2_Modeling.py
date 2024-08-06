import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

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
        Como vimos anteriormente, valores passados são totalmente influentes ao ponto de virada, nessa página iremos mostrar como um modelo
        treinado usando esses valores, pode ser aplicado de forma eficiente para prever se um aluno atingirá ou não o ponto de virada. 🚀
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
        Aqui temos um exemplo de um modelo e de sua performance na previsão do Ponto de Virada dos alunos, 
        na próxima página iremos explorar esse modelo e colocar ele pra uso com dados inputados manualmente. 
        '''
    )

if __name__ == "__main__":
    run()
