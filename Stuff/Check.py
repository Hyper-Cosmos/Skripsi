import pandas as pd

# Nama file CSV yang akan dibaca
csv_filename = 'dataset.csv'

# Membaca file CSV ke dalam dataframe pandas
df = pd.read_csv(csv_filename)

# Menampilkan dataframe
print(df)
