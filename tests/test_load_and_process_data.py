import pytest
import pandas as pd
import sys
import os
sys.path.append('src')
from load_and_process_data import load_data_train, create_dummies

@pytest.fixture
def make_df():
    df = pd.DataFrame(
        data={
            'temp': [17.3, 25.4, 23.3, 26.9, 20.2],
            'weekday': [6, 3, 5, 1, 1],
            'cost': [1.5, 0.3, 1.5, 0.3, 1.0],
            'price': [5.6, 4.9, 7.6, 5.3, 7.2],
            'sales': [173, 196, 207, 241, 227]
        }
    )
    yield df

@pytest.fixture
def save_df(make_df):
    temp_path = 'temp_df.csv'
    make_df.to_csv(temp_path, index=False)

    yield {'data': {'path': temp_path}}

    #delete file
    os.remove(temp_path)


def test_create_dummies(make_df):
    features = ['weekday']
    df_dummy = create_dummies(make_df, features)
    cols = df_dummy.columns
    assert 'weekday_1' in cols
    assert 'weekday_6' in cols
    assert 'weekday' not in cols
    assert list(df_dummy['weekday_1']) == [0, 0, 0, 1, 1]

def test_load_data_train(save_df):
    df = load_data_train(save_df)
    cols = df.columns
    print(cols)
    assert len(df) == 5
    assert len(cols) == 5