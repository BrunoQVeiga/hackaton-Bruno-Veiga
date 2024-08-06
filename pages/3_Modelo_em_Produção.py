import streamlit as st
import pandas as pd
import joblib

# Configurar a página do Streamlit
st.set_page_config(
    page_title="Previsão do Ponto de Virada",
    page_icon="🔮",
)

def run():
    st.write("# Previsão do Ponto de Virada 🔮")

    st.markdown(
        '''
        Nesta página, você pode inserir manualmente os dados de desempenho do aluno para prever se ele atingirá o **Ponto de Virada**.
        Insira os valores nos campos abaixo e clique em "Prever" para ver o resultado. 🚀
        '''
    )

    # Carregar o modelo treinado
    model = joblib.load('gradient_boosting_model.pkl')

    # Carregar o dataset original para pegar os valores máximos
    df = pd.read_csv('Final_Merged_DataFrame.csv')

    # Converter colunas de interesse para numérico
    cols_to_numeric = [
        'PEDRA_2021', 'INDE_2021', 'IAA_2021', 'IEG_2021', 'IPS_2021',
        'IDA_2021', 'IPP_2021', 'IPV_2021', 'IAN_2021'
    ]

    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remover valores NaN para calcular o máximo corretamente
    df.dropna(subset=cols_to_numeric, inplace=True)

    # Valores máximos para cada coluna
    col_max = {
        'INDE_2021': df['INDE_2021'].max(),
        'IAA_2021': df['IAA_2021'].max(),
        'IEG_2021': df['IEG_2021'].max(),
        'IPS_2021': df['IPS_2021'].max(),
        'IDA_2021': df['IDA_2021'].max(),
        'IPP_2021': df['IPP_2021'].max(),
        'IPV_2021': df['IPV_2021'].max(),
        'IAN_2021': df['IAN_2021'].max()
    }

    # Mapeamento para 'PEDRA_2021'
    pedra_mapping = {
        0: 'Quartzo',
        1: 'Ágata',
        2: 'Ametista',
        3: 'Topázio'
    }

    # Definir os parâmetros de entrada com base nos valores min (0) e max calculados
    input_params = {
        'PEDRA_2021': st.selectbox('PEDRA 2021', options=[(0, 'Quartzo'), (1, 'Ágata'), (2, 'Ametista'), (3, 'Topázio')], format_func=lambda x: x[1])[0],
        'INDE_2021': st.number_input('INDE 2021', min_value=0.0, max_value=col_max['INDE_2021'], step=0.01),
        'IAA_2021': st.number_input('IAA 2021', min_value=0.0, max_value=col_max['IAA_2021'], step=0.01),
        'IEG_2021': st.number_input('IEG 2021', min_value=0.0, max_value=col_max['IEG_2021'], step=0.01),
        'IPS_2021': st.number_input('IPS 2021', min_value=0.0, max_value=col_max['IPS_2021'], step=0.01),
        'IDA_2021': st.number_input('IDA 2021', min_value=0.0, max_value=col_max['IDA_2021'], step=0.01),
        'IPP_2021': st.number_input('IPP 2021', min_value=0.0, max_value=col_max['IPP_2021'], step=0.01),
        'IPV_2021': st.number_input('IPV 2021', min_value=0.0, max_value=col_max['IPV_2021'], step=0.01),
        'IAN_2021': st.number_input('IAN 2021', min_value=0.0, max_value=col_max['IAN_2021'], step=0.01)
    }

    # Botão para executar a previsão
    if st.button('Prever'):
        # Converter os parâmetros para um DataFrame
        input_df = pd.DataFrame([input_params])

        # Fazer a previsão
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        # Exibir o resultado da previsão
        if prediction[0] == 1:
            st.success("O aluno provavelmente atingirá o Ponto de Virada! 🌟")
        else:
            st.warning("O aluno provavelmente não atingirá o Ponto de Virada.")

        # Exibir as probabilidades de previsão
        st.write("### Probabilidades de Previsão")
        st.write(f"Probabilidade de Não: {prediction_proba[0][0]:.2%}")
        st.write(f"Probabilidade de Sim: {prediction_proba[0][1]:.2%}")

if __name__ == "__main__":
    run()



