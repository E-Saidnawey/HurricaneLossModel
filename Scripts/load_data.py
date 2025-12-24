import pandas as pd
import os

def load_hurricane_data(path):
    df = pd.read_csv(path)

    df = df[df['year'] >= 1950]

    df_per_year = df.groupby('year').size().reset_index(name='hurricane_count')
    df_per_year['year'] = df_per_year['year'].astype(int)

    return (df, df_per_year)


if __name__ == "__main__":
    path = os.path.join('Data', 'hurdat2_all_events.csv')

    df, df_per_year = load_hurricane_data(path)

    if df:        
        print(f"df loaded successfully: {json_name}.")
        print(df.head())