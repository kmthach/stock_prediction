import json
import pandas as pd
import requests
import streamlit as st
import torch

st.title('Demo Stock Prediction Of 2022')
symbol = st.text_input('Type symbol here!')
N = st.select_slider('Choose top N symbols',[*range(0, 100)])
inputs = {
    'symbol': symbol,
    'num_years': N
}
def call_api():
    res = requests.post(url = 'http://127.0.0.1:5000/predict',data = json.dumps(inputs))
    data = res.json()
    symbol_price = data['symbol_price']
    topN_data = data['topN']
    
    cols = ['symbol']
    for i in range(1, 13):
        cols.append(f'{i} - 2022')

    cols.append('growth rate')
    symbol_df = pd.DataFrame(columns = cols)
 
    symbol_df.loc[0] = [inputs['symbol']] + symbol_price
    
    
    topN_df = pd.DataFrame(columns = cols)
    for i, (symbol, price) in enumerate(topN_data.items()):
        topN_df.loc[i] = [symbol] + price
    

    st.table(symbol_df)
    st.table(topN_df)
st.button('Predict', on_click= call_api)
