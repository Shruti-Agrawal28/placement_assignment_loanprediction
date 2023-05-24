from loan.exception import loanException
from loan.logger import logging
from loan.predictor import ModelResolver
import pandas as pd
from loan.utils import load_object
import os, sys
from datetime import datetime
import numpy as np
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
PREDICTION_DIR = "prediction"


def start_batch_prediction(input_file_path):
    try:
        os.makedirs(PREDICTION_DIR, exist_ok=True)
        logging.info(f"Creating model resolver object")
        model_resolver = ModelResolver(model_registry="saved_models")
        logging.info(f"Reading file :{input_file_path}")
        df = pd.read_csv(input_file_path)
        df.replace({"na": np.NAN}, inplace=True)
        categorical = []
        for col in df.columns:
            if df[col].dtype == "object":
                categorical.append(col)
        for i in categorical:
            df[i] = le.fit_transform(df[i])
        # validation

        logging.info(f"Loading transformer to transform dataset")
        transformer = load_object(file_path=model_resolver.get_latest_transformer_path())

        input_feature_names = list(transformer.feature_names_in_)
        input_arr = transformer.transform(df[input_feature_names])

        logging.info(f"Loading model to make prediction")
        model = load_object(file_path=model_resolver.get_latest_model_path())
        prediction = model.predict(input_arr)

        logging.info(f"Target encoder to convert predicted column into categorical")
        target_encoder = load_object(file_path=model_resolver.get_latest_target_encoder_path())

        df["prediction"] = prediction
       
        prediction_file_name = os.path.basename(input_file_path).replace(".csv",
                                                                         f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
        prediction_file_path = os.path.join(PREDICTION_DIR, prediction_file_name)
        df.to_csv(prediction_file_path, index=False, header=True)
        return prediction_file_path
    except Exception as e:
        raise loanException(e, sys)
