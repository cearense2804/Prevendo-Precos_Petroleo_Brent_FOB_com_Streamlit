import streamlit as st
import pandas as pd
import numpy as nppip
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px
import altair as alt
import joblib 
import pickle 
import time


# Configurar o layout da página
st.set_page_config(layout="wide")
col1, col2, col3 = st.columns([1,8,1])
col4, col5 = st.columns([7,3])
col7, col8 = st.columns(2) 
col9, col10, col11 = st.columns(3) 
col12, col13 = st.columns(2) 
col14, col15,col16 = st.columns(3)
col17, col18 = st.columns(2)   


#upload de csv
dados = pd.read_csv('ipea.csv')
agrupado= pd.read_csv('agrupado.csv')
completo = pd.read_csv('df_completo.csv')



page_bg_img = '''
<style>
    body {
    background-image: (".\imagem_transparente.png");
    background-size: cover;
    }
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

with col1:
    st.write("")
with col2:
    st.title(":orange-background[Prevendo Preços do Petróleo Brent-FOB]⛽")  
with col3:
    st.write("")


with st.container():  
    with col4:
        with st.container(border=True):  
            st.markdown("###### :orange-background[Preços do Petróleo Brent-FOB (US$) de 2000 à 2024]📈")
            st.line_chart(dados, x='Data', y='Preco_Petroleo_Brent_FOB')
    with col5:
        with st.container(border=True):
                st.write(':orange-background[Medidas Estatísticas]')
                minimo  = dados['Preco_Petroleo_Brent_FOB'].min()
                menor = dados['Preco_Petroleo_Brent_FOB'].min()
                min_data = dados[['Data','Preco_Petroleo_Brent_FOB']].loc[dados['Preco_Petroleo_Brent_FOB'] == menor].set_index('Data')
                if st.button("Preço Mínimo (US$)", type="primary"):
                    st.write(f"Preço mínimo: {minimo} ")
                    st.write('Data de registro:',min_data) 
                else:
                    st.write("")


                media = dados['Preco_Petroleo_Brent_FOB'].mean()
                formatted_string = "{:.2f}".format(media)
                if st.button("Preço Médio(US$)", type="primary"): 
                    st.write(f"Preço médio: {formatted_string}")  
                else:
                    st.write("") 

              
                maior= dados['Preco_Petroleo_Brent_FOB'].max()
                max_data = dados[['Data','Preco_Petroleo_Brent_FOB']].loc[dados['Preco_Petroleo_Brent_FOB'] == maior].set_index('Data')
                if st.button('Preço Máximo (US$)', type="primary"):
                    st.write(f"Preço Máximo: {maior}")
                    st.write('Data de registro:',max_data)    
                else:
                    st.write("") 
                
        


#Visualizando os dados
with st.container(border=True):
    col7, col8 = st.columns(2)
    with col7:
        #Subindo e tratando os dados
        st.markdown(':orange-background[Histórico de Preços (US$)]')
        st.dataframe(dados.set_index('Data'), use_container_width=True)

    
    with col8:
        # Exibindo a base de dados
        st.markdown(':orange-background[Médias de Preços por Ano (US$)]')
        st.bar_chart(agrupado.set_index('ano'),use_container_width=True)
        
st.write(':orange-background[Diferença Percentual de Preços Entre os Dias Subsequentes(%)]')
fig = plt.figure(figsize=(15, 6))
values =completo[ 'Diferenca(%)'].iloc[-180:] 
datas = completo[ 'Data'].iloc[-180:] 
sns.barplot(data=completo.sort_values('Data', ascending=False),x=datas, y=values, color='dodgerblue')
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
plt.gcf().autofmt_xdate()
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.grid(color='lightgrey')
st.pyplot(fig)



#Predições

st.subheader(":orange-background[Previsão de  Preços] 📈", divider="orange")
st.write('###### Prevendo os preços do Petróleo Brent-FOB(US$) com algoritmo de machine learning:')
num = st.slider("Dias a prever: ", 1, 10,5)



dados = pd.read_csv('ipea.csv')
dados['Data'] = pd.to_datetime(dados['Data'])
for lag in range(1, 3):  # Criou atraso de 1 dia nesse lag
    dados[f'Preco_lag_{lag}'] = dados['Preco_Petroleo_Brent_FOB'].shift(lag)
dados = dados.dropna()

X = dados[['Preco_lag_1','Preco_lag_2']].values
y = dados['Preco_Petroleo_Brent_FOB'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False, random_state=42)
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.2, max_depth=6, random_state=42,loss='squared_error')
model.fit(X_train, y_train)
previsao =  model.predict(X_test)

mse = mean_squared_error(y_test, previsao)
mae = mean_absolute_error(y_test, previsao)
r2 = r2_score(y_test, previsao)

#prevendo novas datas
ultima_data = X[-1].reshape(1, -1)
pred_futuro = []

for _ in range(num):  # para cada dia da próxima semana
    pred_dia_futuro = model.predict(ultima_data)[0]
    pred_futuro.append(pred_dia_futuro)
    ultima_data = np.roll(ultima_data, -1)
    ultima_data[0, -1] = pred_dia_futuro

# As datas correspondentes à próxima semana
prox_data = pd.date_range(dados['Data'].iloc[-1], periods=(num+1), freq='B')[1:]
  
# Selecionar os dados da semana atual (últimos 7 dias do dataset)
datas_sem_atual = dados['Data'].iloc[-num:]
preco_sem_atual = dados['Preco_Petroleo_Brent_FOB'].iloc[-num:]


#barra de progresso
mensagem = "Operação em progresso. Por favor, aguarde."
barra = st.progress(0, text=mensagem)
for i in range(100):
    time.sleep(0.01)
    barra.progress(i+1, text=mensagem)
time.sleep(1)
barra.empty()

# Plotar os preços reais da semana atual e as previsões para a próxima semana
col12, col13 = st.columns([3,7])    
with st.container():

    with col12:
        st.container()
        st.caption(f'### ☞ Preços Previstos para {num} dias em US$')
        df = pd.DataFrame()
        df['Data'] = prox_data.values
        df['Preco_Previsto'] = pred_futuro
        st.dataframe(df)
    with col13:
        fig = plt.figure(figsize=(10, 5))
        plt.plot(datas_sem_atual, preco_sem_atual, color='royalblue',marker='o',linestyle='-',label='Preços Atuais')
        plt.plot(prox_data, pred_futuro, color='tomato',marker='o',linestyle='dashed',label='Previsões para a Próxima Semana')
        # Formatar o eixo x para exibir datas
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        #plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.gcf().autofmt_xdate()  # Ajustar formato das datas para evitar sobreposição
        plt.xlabel('Data', fontsize=8)
        plt.ylabel('Preço (US$)', fontsize=8)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.title('Preços Reais e Previsões', fontsize=11)
        plt.legend(loc='best', fontsize=9)
        plt.grid(True)
        st.pyplot(fig)
 

#Visualizando as Estatísticas
st.subheader("", divider="orange")
col14, col15,col16 = st.columns(3)    
with st.container():
    col14, col15,col16 = st.columns([3,1,6])   
    with col14:
        st.markdown("##### Fontes: ") 
        st.write('       ☞ Instituto de Pesquisa Econômica Aplicada-IPEA')
        st.write('       ☞ YFinance')
    with col15:
        st.write("")
        st.write("")
        st.link_button("Site do IPEA", "http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view",type="secondary")
    with col16:
        st.write("")

st.markdown('##### Links:')
with st.container():
    col17, col18 = st.columns([3,7])   
    with col17:   
        st.write("☞ Github:")
        st.write("")
        st.write("☞ Streamlit App: ")       
    with col18:
        st.link_button("https://github.com/cearense2804/Prevendo-Precos_Petroleo_Brent_FOB_com_Streamlit/blob/main/app.py", "https://github.com/cearense2804/Prevendo-Precos_Petroleo_Brent_FOB_com_Streamlit/blob/main/app.py",type="secondary")
        st.link_button("Streamlit App", "https://prevendo-precospetroleobrentfobcomapp-fffmbxouavt3ffjtinxna7.streamlit.app/",type="secondary")
   


    