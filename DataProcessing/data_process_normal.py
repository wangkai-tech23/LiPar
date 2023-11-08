import pandas as pd
import warnings
warnings.filterwarnings("ignore")

file_path = '../../Car-Hacking Dataset/normal.csv'

# Read dataset
df = pd.read_csv(file_path, header=None, usecols=[2, 4, 5, 6, 7, 8, 9, 10, 11])

# Add fieldnames
df.columns = ['CAN_ID', 'Data[0]', 'Data[1]', 'Data[2]', 'Data[3]',
              'Data[4]', 'Data[5]', 'Data[6]', 'Data[7]']
# print(df.Label.value_counts())

# Data cleaning
rule = {'CAN_ID': '0000', 'Data[0]': '00', 'Data[1]': '00', 'Data[2]': '00', 'Data[3]': '00',
        'Data[4]': '00', 'Data[5]': '00', 'Data[6]': '00', 'Data[7]': '00'}
df.fillna(rule, inplace=True)


# Add label column
df.insert(loc=len(df.columns), column='Label', value='R')

# Save to new file
df.to_csv('./normal_dataset.csv', index=False)

