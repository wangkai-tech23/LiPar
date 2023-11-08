import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# change the file name to reprocess different types of attack data
file_path = '../Car-Hacking Dataset/Spoofing_the_RPM_gauge_dataset.csv'

# Read dataset
df = pd.read_csv(file_path, header=None, usecols=[1, 3, 4, 5, 6, 7, 8, 9, 10, 11])

# Add fieldnames
df.columns = ['CAN_ID', 'Data[0]', 'Data[1]', 'Data[2]', 'Data[3]',
              'Data[4]', 'Data[5]', 'Data[6]', 'Data[7]', 'Label']
# print(df.Label.value_counts())

# Data cleaning
rule = {'CAN_ID': '0000', 'Data[0]': '00', 'Data[1]': '00', 'Data[2]': '00', 'Data[3]': '00',
        'Data[4]': '00', 'Data[5]': '00', 'Data[6]': '00', 'Data[7]': '00', 'Label': 'R'}
df.fillna(rule, inplace=True)
df.loc[df["Data[1]"].isin(['R']), "Data[1]"] = '00'
df.loc[df["Data[2]"].isin(['R']), "Data[2]"] = '00'
df.loc[df["Data[3]"].isin(['R']), "Data[3]"] = '00'
df.loc[df["Data[4]"].isin(['R']), "Data[4]"] = '00'
df.loc[df["Data[5]"].isin(['R']), "Data[5]"] = '00'
df.loc[df["Data[6]"].isin(['R']), "Data[6]"] = '00'
df.loc[df["Data[7]"].isin(['R']), "Data[7]"] = '00'


# Convert hex to dec
df.CAN_ID = df['CAN_ID'].apply(int, base=16)
df['Data[0]'] = df['Data[0]'].apply(int, base=16)
df['Data[1]'] = df['Data[1]'].apply(int, base=16)
df['Data[2]'] = df['Data[2]'].apply(int, base=16)
df['Data[3]'] = df['Data[3]'].apply(int, base=16)
df['Data[4]'] = df['Data[4]'].apply(int, base=16)
df['Data[5]'] = df['Data[5]'].apply(int, base=16)
df['Data[6]'] = df['Data[6]'].apply(int, base=16)
df['Data[7]'] = df['Data[7]'].apply(int, base=16)

# Save to new file
df.to_csv('./DataProcess/RPM_dataset.csv', index=False)

