# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

"""
Descrição: Este programa usa uma rede neural artificial recorrente chamada Long Short Term Memory (LSTM)
para prever o preço de fechamernto de ações de uma corporação (no exemplo, Petrobras) usando dados dos ultimos 8 anos
"""

#Import libraries
import math
import pandas_datareader as web #Para a obtenção de bases de dados com base em fontes pre-definidas
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler   #Biblioteca geral para tratamento de dados e ML
from keras.models import Sequential              # Sequential, Dense e LSTM são as bibiotecas de redes neurais
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#Obtenção das bases de dados de ações via DataReader
df = web.DataReader('PETR4.SA', data_source='yahoo', start='2012-01-01', end='2020-04-09')
#Como teste, exibir a variavel
#df

#Teste: obter forma da variavel (dimensões da matriz)
#df.shape

#Visualização com matplotlib
plt.figure(figsize=(16, 8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price BR($)', fontsize=18)
plt.show()


# criar um dataframe separado apenas com a coluna Closing (fechamento), visto que é só os preços de fechamento que interessam
data = df.filter(['Close'])
#Converter para array numpy para que possa ser trabalhado pelas funções seguintes
dataset = data.values

#Separação dos dados em grupos de treino e grupos de teste. Na proporção de, mais ou menos 80/20 pct
#Get the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * .8)

#Fazer a escalagem dos dados, deixando eles em um espaço entre 0 e 1.
#Não essencial para todos os problemas, mas ajuda na maioria, para que os cálculos sejam mais rápidos
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)


#Criação dos dados de treino escalados
train_data = scaled_data[0:training_data_len, :]
#Split the data into x_train and y_trin
x_train = [] #independent variables
y_train = [] #target variables

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

#Não descomentar, foi só pra teste    
#    if i <= 61:                   
#        print(x_train)
#        print(y_train)
#        print()

#Converter as arrays x_train e y_train para numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#Reformar as variáveis de treino

#A rede neural LSTM espera receber datasets de dados em 3D (Arrays de 3 colunas)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Montando o modelo / adicionando duas 'camadas' de 50 neuronios da rede tipo LSTM
#Adiciona-se mais duas camadas de neurionios do tipo de rede neural Dense, sendo a última camada o 'neuronio de saída'
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#Compilar o modelo / usado para fazer a medida erro do modelo
model.compile(optimizer='adam', loss='mean_squared_error')

#Treinando o modelo
model.fit(x_train, y_train, batch_size=1, epochs=1)


#Criando o dataset de teste

#Cria-se nova array contendo valores escalados correspondentes a mais ou menos 20pct. da base original
test_data = scaled_data[training_data_len - 60:, :]
#Create the datasets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])


#Convertendo datasets em array numpy
x_test = np.array(x_test)

#Reformar o dataset 
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#Fazer previsões com base no modelo
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

    
##Verificar a precisão do modelo por meio do root mean square error ('raiz do erro médio ao quadrado')
"""
O objetivo é que o mean square error seja o mais próximo de zero quanto possível.
Isso indica que a taxa de erro entre as previsões do modelo e os reais valores do grupo de teste é baixa,
ISSO signficando que o modelo está fazendo boas previsões (matematicamente)
"""

rsme = np.sqrt(np.mean(predictions - y_test) ** 2)
#print value to check

#Vizualizando os resultados
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

#Vizualiação dos dadosreais contra o previstos
plt.figure(figsize=(16, 8))
plt.title('Model Evaluation')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Cole Price BR$', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

#Show the actual price and predicted prices

#type valid


#Verificação de previsão contra valores reais

stock_quote = web.DataReader('PETR4.SA', data_source='yahoo', start='2012-01-01', end='2020-04-09') #Captura o dado valor de fechamento de algum dia

#Coloca em um novo dataframe, pegando a coluna que interessa (fechamento)
new_df = stock_quote.filter(['Close'])
# Pega os últimos 60 dias dos valores de fechamento e converte para array
last_60_days = new_df[-60:].values
#Escalar os valores
last_sixty_days_scaled = scaler.transform(last_60_days)
#Create an empty list
X_test = []
#Adicionar os valores escalados à lisa criada acima
X_test.append(last_sixty_days_scaled)
#Converter para array numpy
X_test = np.array(X_test)
#Colocar em formato 3D
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#Fazer a previsão usando o modelo treinado
pred_price = model.predict(X_test)
#Desfazer a scalade de valores de volta ara o formato original, de forma a poder ler os resultados
pred_price = scaler.inverse_transform(pred_price)

#pred_price será o valor previsto pra o preço de fechamento da ação, no dia final que você colocou na linha 151

#print the predicted price (pred_price) [closing] for the following day, according to the model


#Confira a precisão puxando o valor real
stock_quote2 = web.DataReader('PETR4.SA', data_source='yahoo', start='2020-04-09', end='2020-04-09') 

