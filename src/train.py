from prophet import Prophet
import pickle


def train_model(multi_data_train, save_model=False):
    """Train model

    Args:
        multi_data_train (dataframe): train data

    Returns:
        model: trained model
    """
    model = Prophet()
    for column in multi_data_train.columns[2:]:
        print(column)
        model.add_regressor(column)

    model.fit(multi_data_train)

    if save_model:
        with open('/models/trained_model.pkl', 'wb') as file:
            pickle.dump(model, file)

    return model