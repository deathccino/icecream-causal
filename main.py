from src.utils import load_config
from src.load_and_process_data import load_data_test, create_dummies, save_df
from src.model_training import train_estimator

def main():
    #load the yaml configuration file
    config = load_config()

    df = load_data_test(config)
    df_prc = create_dummies(df, ['weekday'])
    print(df_prc.columns)
    filename = 'ice_cream_prc.csv'
    save_df(config, df_prc, filename)
    print('file saved succesfully')

    m1 = train_estimator(config, filename)
    print(m1)


if __name__ == "__main__":
    main()
