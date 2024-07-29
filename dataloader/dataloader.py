import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path, encoding='latin-1')
    df['Label'] = df['sentiment'].apply(lambda x: '1' if x == 'Positive' else ('0' if x == 'Negative' else None))
    df = df[df['Label'].isin(['1', '0'])]  # Filter out rows where Label is None
    data = df[['review', 'Label']]
    return data
