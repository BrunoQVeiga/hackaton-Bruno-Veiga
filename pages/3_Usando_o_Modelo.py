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
    model_regression = joblib.load('gradient_boosting_model_regression.pkl')

    # Carregar o dataset original para pegar os valores máximos
    df = pd.read_csv('Final_Merged_DataFrame.csv')

    # Converter colunas de interesse para numérico
    cols_to_numeric = [
        'PEDRA_2021', 'INDE_2021', 'IAA_2021', 'IEG_2021', 'IPS_2021',
        'IDA_2021', 'IPP_2021', 'IPV_2021', 'IAN_2021'
    ]

    performance_columns_class = [
        'PEDRA_2021', 'IAA_2021', 'IEG_2021', 'IPS_2021', 
        'IDA_2021', 'IPP_2021', 'IPV_2021']
    
    performance_columns_regress = [
    'INDE_2021', 'IAA_2021', 'IEG_2021', 'IAN_2021', 
    'IDA_2021', 'IPP_2021', 'IPV_2021']

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
        'PEDRA_2021': st.selectbox('PEDRA', options=[(0, 'Quartzo'), (1, 'Ágata'), (2, 'Ametista'), (3, 'Topázio')], format_func=lambda x: x[1])[0],
        'INDE_2021': st.number_input('INDE', min_value=0.0, max_value=col_max['INDE_2021'], step=0.01),
        'IAA_2021': st.number_input('IAA', min_value=0.0, max_value=col_max['IAA_2021'], step=0.01),
        'IEG_2021': st.number_input('IEG', min_value=0.0, max_value=col_max['IEG_2021'], step=0.01),
        'IPS_2021': st.number_input('IPS', min_value=0.0, max_value=col_max['IPS_2021'], step=0.01),
        'IDA_2021': st.number_input('IDA', min_value=0.0, max_value=col_max['IDA_2021'], step=0.01),
        'IPP_2021': st.number_input('IPP', min_value=0.0, max_value=col_max['IPP_2021'], step=0.01),
        'IPV_2021': st.number_input('IPV', min_value=0.0, max_value=col_max['IPV_2021'], step=0.01),
        'IAN_2021': st.number_input('IAN', min_value=0.0, max_value=col_max['IAN_2021'], step=0.01)
    }

    # Botão para executar a previsão
    if st.button('Prever'):
        # Converter os parâmetros para um DataFrame
        input_df = pd.DataFrame([input_params])

        # Fazer a previsão
        prediction = model.predict(input_df[performance_columns_class])
        prediction_proba = model.predict_proba(input_df[performance_columns_class])

        # Exibir o resultado da previsão
        if prediction[0] == 1:
            st.success("O aluno provavelmente atingirá o Ponto de Virada! 🌟")
        else:
            st.warning("O aluno provavelmente não atingirá o Ponto de Virada.")

        # Exibir as probabilidades de previsão
        st.write("### Probabilidades de Previsão")
        st.write(f"Probabilidade de Não: {prediction_proba[0][0]:.2%}")
        st.write(f"Probabilidade de Sim: {prediction_proba[0][1]:.2%}")

        # Fazer a previsão
        prediction_inde = model_regression.predict(input_df[performance_columns_regress])

        inde_value = prediction_inde[0]
        if 2.405 <= inde_value < 5.506:
            pedra = "Quartzo"
        elif 5.506 <= inde_value < 6.868:
            pedra = "Ágata"
        elif 6.868 <= inde_value < 8.230:
            pedra = "Ametista"
        elif 8.230 <= inde_value <= 9.294:
            pedra = "Topázio"
        else:
            pedra = "Fora dos intervalos definidos"

        # Exibir o resultado da previsão
        st.write(f"### Valor Previsto de INDE: {inde_value:.2f}")
        if inde_value > 6.868:
            st.success(f"O aluno está previso para ser classificado na pedra: {pedra}")
        else: 
            st.warning(f"O aluno está previso para ser classificado na pedra: {pedra}")
        
if __name__ == "__main__":
    run()



