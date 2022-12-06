import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load do DataFrame
df = pd.read_csv('Variaveis\dados.csv', delimiter = ',', header = 0)
df['mes_ref'] = pd.to_datetime(df['mes_ref'])
df.set_index('mes_ref', inplace= True)

#Mapa de calor Correlação de Pearson
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot = True, cmap= "RdYlGn")
plt.title('Correlação de Pearson',size=15)
plt.show()

