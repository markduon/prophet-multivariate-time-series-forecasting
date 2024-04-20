import pickle
import pandas as pd


def evaluate_model(df_multi, multi_data_test):
    """Evaluation function

    Args:
        df_multi (dataframe): _description_
        multi_data_test (dataframe): _description_

    Returns:
        final_result: result of prediction
    """
    with open('/models/trained_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    future_multi = loaded_model.make_future_dataframe(periods=len(multi_data_test))
    future_multi = pd.merge(future_multi, df_multi.drop("y", axis=1),how='left', on='ds')
    df_clean = future_multi.dropna()
    forecast_multi = loaded_model.predict(df_clean)
    y_pred_df = forecast_multi[['ds', 'yhat']]
    result = pd.merge(multi_data_test[['ds', 'y']], y_pred_df, how='inner')
    final_result = result.set_index('ds').dropna()
    return final_result