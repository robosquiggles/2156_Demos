import os
import pickle, dill, inspect
import pandas as pd
import datetime as time

import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression

import numpy as np

required_columns = ["Timestamp", "Model Name", "Model Path", "Model Object"]

def pickle_models_df(df:pd.DataFrame, directory:str, df_file_name:str):
    """
    Pickles the given `df:pandas.DataFrame` of Model information to the given `df_file_name` in the given `directory`.
    If the `Model Path` for the model is empty for any given Model (row), the model iteslt is saved to the `directory`
    under a unique file name, and the file name is added to the `Model Path` for the model (before pickling).

    Parameters:
        df: pd.DataFrame of the models containing the `required_columns`
        directory: str, the directory where the models and DataFrame will be saved
        df_file_name: str, the name of the file where the DataFrame will be pickled
    """
    if not all(column in df.columns for column in required_columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
    
    for index, row in df.iterrows():
        if pd.isna(row["Model Path"]) or row["Model Path"] == "":
            class_file_name = f"model_{index}.pkl"
            class_path = f"{directory}/{class_file_name}"
            with open(class_path, "wb") as f:
                dill.dump(row["Model Object"], f)
            df.at[index, "Model Path"] = class_path

    df = df.drop(columns=["Model Object"]) # Remove the row before pickling becuase it doesn't unpickle nicely. We have the model saved to reload from.

    df_path = f"{directory}/{df_file_name}"
    with open(df_path, 'wb') as df_file:
        pickle.dump(df, df_file)

    print("Pickle Success!!")


def depickle_models_from_file(path:str, load_models = False):
    """
    De-pickles models with details in the given path. Models must have been pickled using the `pickle_models_df` function!!

    Parameters:
        path: the location from which to unpickle the models
    Returns:
        models_df: a `pd.DataFrame` with all of the models pickled at the given location.
    """

    with open(path, 'rb') as df_file:
        models_df = pickle.load(df_file)
        models_df["Model Object"] = None # Adding an empty row to load the models back into

    if load_models:
        load_predictors(models_df)

    print("De-pickle Success!!")

    return models_df


def load_predictors(models_df:pd.DataFrame):
    for index, row in models_df.iterrows():
        if not pd.isna(row["Model Path"]) and row["Model Path"] != "":
            model_path = row["Model Path"]
            class_path = row["Model Class Path"]
            with open(class_path, "rb") as f:
                model = dill.load(f)
            model.load_state_dict(torch.load(model_path))
            models_df.at[index, "Model Object"] =  model

    print("Models Loaded!!")
    return models_df


def record_model_to_df(df:pd.DataFrame, model_name:str, model:nn.Module|LinearRegression, train_r2:float, test_r2:float):
    """
    Records a new model's information into the DataFrame.

    Parameters:
        df: pd.DataFrame, the DataFrame to record the model information into
        model_name: str, the name/type of the model
        model: the model object itself
        train_r2: float, the R2 score on the training data
        test_r2: float, the R2 score on the test data

    Returns:
        df: pd.DataFrame, the updated DataFrame with the new model information
    """
    df.loc[df.shape[0]] = [time.datetime.now(),model_name,"","",model]

    return df


def remove_model(index:int, directory:str, file_name:str):
    """

    **!! USE WITH CAUTION !!**

    Removes the model at the given index from the DataFrame *and deletes its corresponding model files.*

    Parameters:
        index: int, the index of the model to remove
        directory: the folder from which to get the file
        file_name: the name of the file from which to remove

    Returns:
        df: pd.DataFrame, the updated DataFrame with the model removed
    """
    df = depickle_models_from_file(f"{directory}/{file_name}",load_models=False)

    if index not in df.index:
        raise IndexError(f"Index {index} is not in the DataFrame")

    model_path = df.at[index, "Model Path"]
    class_path = df.at[index, "Model Class Path"]

    if model_path and os.path.exists(model_path):
        os.remove(model_path)
        print("Removed the model:", model_path)
    if class_path and os.path.exists(class_path):
        os.remove(class_path)
        print("Removed the object:", class_path)

    df = df.drop(index)

    pickle_models_df(df, directory, file_name)

    return df