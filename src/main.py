from train import train_model
from dataset import process_data
from evaluate import evaluate_model


if __name__ == "__main__":
    datatrain_path = "data/train.csv"
    datatest_path = "data/test.csv"
    df_multi, multi_data_train, multi_data_test = process_data(datatrain_path, datatest_path)
    model = train_model(multi_data_train, save_model=True)
    result = evaluate_model(df_multi, multi_data_test)
    print(result)
