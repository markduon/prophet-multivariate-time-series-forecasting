import pandas as pd


def process_data(datatrain_path, datatest_path):
    """function for processing data before fitting into model

    Args:
        datatrain_path (str): trainset path
        datatest_path (str): testset path

    Returns:
        df_multi, multi_data_train, multi_data_test: processed data
    """
    df_train = pd.read_csv(datatrain_path, index_col='timestamp', parse_dates=True)
    df_test = pd.read_csv(datatest_path, index_col='timestamp', parse_dates=True)
    df = pd.concat([df_train, df_test])
    df_multi = df.reset_index()
    df_multi = df_multi.rename(columns={'timestamp': 'ds', 'price': 'y'})

    multi_data_train = df_train.reset_index()
    multi_data_train = multi_data_train.rename(columns={'timestamp': 'ds', 'price': 'y'})

    multi_data_test = df_test.reset_index()
    multi_data_test = multi_data_test.rename(columns={'timestamp': 'ds', 'price': 'y'})

    return df_multi, multi_data_train, multi_data_test
