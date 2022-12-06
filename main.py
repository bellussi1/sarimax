import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pmdarima as pm

# Estilização do Plot

# DataFrame Exp e Imp do BB + Taxas (01/2014 - 09/2022)
df = pd.read_csv('Variaveis\dados2.csv', delimiter = ',', header = 0)
df['mes_ref'] = pd.to_datetime(df['mes_ref'])
df.set_index('mes_ref', inplace= True)

# DataFrame de projeção do BB Base (10/2022 - 10/2024)
df_proj = pd.read_csv('Variaveis\projecao.csv', delimiter = ',', header = 0)
df_proj['mes_ref'] = pd.to_datetime(df_proj['mes_ref'])
df_proj.set_index('mes_ref', inplace= True)

# DataFrame de projeção do BB Otimista (10/2022 - 10/2024)
df_projg = pd.read_csv('Variaveis\projecaogood.csv', delimiter = ',', header = 0)
df_projg['mes_ref'] = pd.to_datetime(df_projg['mes_ref'])
df_projg.set_index('mes_ref', inplace= True)

# DataFrame de projeção do BB Pessimista (10/2022 - 10/2024)
df_projb = pd.read_csv('Variaveis\projecaobad.csv', delimiter = ',', header = 0)
df_projb['mes_ref'] = pd.to_datetime(df_projb['mes_ref'])
df_projb.set_index('mes_ref', inplace= True)

# Series para Np-Array e dropando colunas de endógenas
exog = df.drop(columns=['volume_exp', 'volume_imp']).to_numpy().reshape(-1,6)

# Estruturando o modelo SARIMAX 
model = pm.auto_arima(y = df['volume_imp'], X = exog,
                         start_p = 0, d=1, start_q=0, D=1, start_Q = 0, max_P=5, max_D=5,
                         max_Q = 5, m=12, seasonal=True, start_P=0, max_order=4, 
                         test='adf', error_action='warn', suppress_warnings=True,
                         stepwise=True, trace=True, n_fits = 50, random_state = 42 )

# Numero de meses a ser previsto
n_periods = 12

# Separando os dados treinado e intervalo de confidence
fitted, confint = model.predict(n_periods=n_periods, X = df_projb, return_conf_int=True)
# Definindo o index
index_of_fc = pd.date_range(df['volume_imp'].index[-1], periods = n_periods, freq='MS')
# Transformando os dados para Series
fitted_series = pd.Series(fitted, index=index_of_fc) # Dados preditados 
lower_series = pd.Series(confint[:, 0], index=index_of_fc) # Confidence Interval
upper_series = pd.Series(confint[:, 1], index=index_of_fc) # Confidence Interval

# Plot dos dados reais
plt.style.use('seaborn-v0_8')

plt.plot(df['volume_imp'])
# Plot de previsão futura
plt.style.use('seaborn-v0_8')

plt.plot(fitted_series, color='green')
# Plot da area de confiança (Margem de erro)
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)
# Titulo do plot
plt.ylabel("Valor em Milhão(R$)")
plt.legend(['Valores Reais', 'Valores Previstos', 'Intervalo de Confiança'], loc = 'lower left' )
plt.show()

# Summary do modelo
print(model.summary())
print(fitted_series)
# Plot do diagnóstico de resíduos
model.plot_diagnostics(figsize=(7,5))
plt.show()
