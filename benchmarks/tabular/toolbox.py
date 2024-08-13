"""
Coauthors: Michael Ainsworth
           Jayanta Dey
           Haoyin Xu
           Noga Mudrik
"""


import numpy as np
import pandas as pd
from random import sample
from sklearn.ensemble import RandomForestClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import xgboost as xgb
import openml
import json
from os.path import exists
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings("ignore")

def convert_ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(i) for i in obj]
    else:
        return obj

def random_sample_new(
    X_train,
    y_train,
    training_sample_sizes,
    seed_rand=0,
    min_rep_per_class=1,
):
    """
    Peforms multiclass predictions for a random forest classifier with fixed total samples.
    min_rep_per_class = minimal number of train samples for a specific label
    """
    np.random.seed(seed_rand)
    training_sample_sizes = sorted(training_sample_sizes)
    num_classes = len(np.unique(y_train))
    classes_spec = np.unique(y_train)
    previous_partitions_len = np.zeros(num_classes)
    previous_inds = {class_val: [] for class_val in classes_spec}
    final_inds = []
    # Check that the training sample size is big enough to include all class representations
    if np.floor(np.min(training_sample_sizes) / num_classes) < min_rep_per_class:
        raise ValueError(
            "Not enough samples for each class, decreasing number of classes"
        )
    for samp_size_count, samp_size in enumerate(training_sample_sizes):
        partitions = np.array_split(np.array(range(samp_size)), num_classes)
        partitions_real = [
            len(part_new) - previous_partitions_len[class_c]
            for class_c, part_new in partitions
        ]  # partitions_real is the number of additional samples we have to take
        indices_classes_addition_all = (
            []
        )  # For each samples size = what indices are taken for all classes together
        for class_count, class_val in enumerate(classes_spec):
            indices_class = np.argwhere(np.array(y_train) == class_val).T[0]
            indices_class_original = [
                ind_class
                for ind_class in indices_class
                if ind_class not in previous_inds[class_val]
            ]
            np.random.shuffle(indices_class_original)
            if partitions_real[class_count] <= len(indices_class_original):
                indices_class_addition = indices_class_original[
                    : partitions_real[class_count]
                ]
                previous_inds[class_val].extend(indices_class_addition)
                indices_classes_addition_all.extend(indices_class_addition)

            else:
                raise ValueError(
                    "Class %s does not have enough samples" % str(class_val)
                )
        if final_inds:
            indices_prev = final_inds[-1].copy()
        else:
            indices_prev = []
        indices_class_addtion_and_prev = indices_prev + indices_class_addition
        final_inds.append(indices_class_addtion_and_prev)
        previous_partitions_len = [len(parti) for parti in partitions]

    return final_inds


# def sample_large_datasets(X_data, y_data, max_size=10000):
#     """
#     For large datasets with over 10000 samples, resample the data to only include
#     10000 random samples.
#     """
#     inds = [i for i in range(X_data.shape[0])]
#     fin = sorted(sample(inds, max_size))
#     return X_data[fin], y_data[fin]

def sample_large_datasets(X_data, y_data, max_size=10000):
    """
    For large datasets with over 10000 samples, resample the data to only include
    10000 random samples. Modified by Ziyan.
    """
    inds = np.random.choice(X_data.index, size=max_size, replace=False)
    return X_data.iloc[inds], y_data.iloc[inds]


def save_best_parameters(
    save_methods, save_methods_rewrite, path_save, best_parameters
):
    """
    Save Hyperparameters
    """
    if exists(path_save + ".json") and save_methods_rewrite["json"] == 0:
        with open(path_save + ".json", "r") as json_file:
            dictionary = json.load(json_file)
        best_parameters_to_save = {**dictionary, **best_parameters}
    else:
        best_parameters_to_save = convert_ndarray_to_list(best_parameters)
    with open(path_save + ".json", "w") as fp:
        json.dump(best_parameters_to_save, fp)


def open_data(path, format_file):
    """
    Open existing data
    """
    dictionary = json.load(path + ".json")
    return dictionary


def create_parameters(model_name, varargin, p=None):
    """
    Functions to calculate model performance and parameters.
    """
    if model_name == "DN":
        parameters = {
            "hidden_layer_sizes": varargin["node_range"],
            "alpha": varargin["alpha_range_nn"],
        }
    elif model_name == "RF":

        parameters = {
            "max_features": list(
                set(
                    [
                        round(p / 4),
                        round(np.sqrt(p)),
                        round(p / 3),
                        round(p / 1.5),
                        round(p),
                    ]
                )
            )
        }
    elif model_name == "GBDT":
        parameters = {
            "learning_rate": varargin["alpha_range_nn"],
            "subsample": varargin["subsample"],
        }
    else:
        raise ValueError(
            "Model name is invalid. Please check the keys of models_to_run"
        )
    return parameters


def do_calcs_per_model(
    all_parameters,
    best_parameters,
    all_params,
    model_name,
    varargin,
    classifiers,
    X,
    y,
    dataset_index,
    train_indices,
    val_indices,
    p=None,
    varCV=None,
):
    """
    find best parameters for given sample_size, dataset_index, model_name
    """
    model = classifiers[model_name]
    varCVmodel = varCV[model_name]
    parameters = create_parameters(model_name, varargin, p)
    clf = RandomizedSearchCV(
        model,
        parameters,
        n_jobs=varCVmodel["n_jobs"],
        cv=[(train_indices, val_indices)],
        verbose=varCVmodel["verbose"],
    )
    clf.fit(X, y)
    all_parameters[model_name][dataset_index] = parameters
    best_parameters[model_name][dataset_index] = clf.best_params_
    all_params[model_name][dataset_index] = clf.cv_results_["params"]
    train_accuracy = clf.score(X[train_indices], y[train_indices])
    val_accuracy = clf.score(X[val_indices], y[val_indices])
    return all_parameters, best_parameters, all_params, train_accuracy, val_accuracy


def load_cc18():
    """
    Import datasets from OpenML-CC18 dataset suite
    """
    X_data_list = []
    y_data_list = []
    dataset_name = []

    for data_id in openml.study.get_suite("OpenML-CC18").data:
        try:
            successfully_loaded = True
            dataset = openml.datasets.get_dataset(data_id, download_data=True, download_qualities=True, download_features_meta_data=True)
            dataset_name.append(dataset.name)
            X, y, is_categorical, _ = dataset.get_data(
                dataset_format="dataframe", target=dataset.default_target_attribute
            )
            _, y = np.unique(y, return_inverse=True)
            X = X.to_numpy() 
            X = np.nan_to_num(X)
        except TypeError:
            successfully_loaded = False
        if successfully_loaded and np.shape(X)[1] > 0:
            X_data_list.append(X)
            y_data_list.append(y)

    return X_data_list, y_data_list, dataset_name

# def import_datasets


def return_to_default():
    """
    function to return to default values
    """
    nodes_combination = [20, 100, 180, 260, 340, 400]
    dataset_indices_max = 72
    max_shape_to_run = 10000
    alpha_range_nn = [0.0001, 0.001, 0.01, 0.1]
    subsample = [0.5, 0.8, 1.0]
    models_to_run = {"RF": 1, "DN": 1, "GBDT": 1}
    return (
        nodes_combination,
        dataset_indices_max,
        max_shape_to_run,
        models_to_run,
        subsample,
        alpha_range_nn,
    )


def save_vars_to_dict(
    classifiers,
    varargin,
    reload_data=False,
    nodes_combination=[20],
    dataset_indices_max=2,
    max_shape_to_run=10000,
    alpha_range_nn=[0.1],
    subsample=[1.0],
    path_to_save="metrics/dict_parameters_test.json",
    shape_2_evolution=1,
    shape_2_all_sample_sizes=8,
):
    """
    save variables to dict
    Function to save variables to dict for later use
    """
    dict_to_save = {
        "reload_data": reload_data,
        "nodes_combination": nodes_combination,
        "dataset_indices_max": dataset_indices_max,
        "max_shape_to_run": max_shape_to_run,
        "alpha_range_nn": alpha_range_nn,
        "subsample": subsample,
        "classifiers_names": tuple(classifiers.keys()),
        "varargin": varargin,
        "shape_2_evolution": shape_2_evolution,
        "shape_2_all_sample_sizes": shape_2_all_sample_sizes,
    }

    with open(path_to_save, "w") as fp:
        json.dump(dict_to_save, fp)


def model_define(model_name, best_params_dict, dataset):
    """ """
    if model_name == "RF":
        model = RandomForestClassifier(
            **best_params_dict[model_name][dataset], n_estimators=500, n_jobs=-1
        )
    elif model_name == "DN":
        model = TabNetClassifier(**best_params_dict[model_name][dataset])
    elif model_name == "GBDT":
        model = xgb.XGBClassifier(
            booster="gbtree",
            base_score=0.5,
            **best_params_dict[model_name][dataset],
            n_estimators=500
        )
    else:
        raise ValueError("Invalid Model Name")
    return model


def mod_dict(res_dict, type_to_compare):
    tot_dict = {}
    for model_name in res_dict.keys():
        modified_subdict = {}
        for dataset in res_dict[model_name].keys():
            data_set_dict = res_dict[model_name][dataset]
            data_set_dict = {
                key: val
                for key, val in data_set_dict.items()
                if isinstance(val, type_to_compare)
            }
            modified_subdict[dataset] = data_set_dict
        tot_dict[model_name] = modified_subdict
    return tot_dict


def read_params_dict_json(path="metrics/cc18_all_parameters_test", type_file=".json"):
    """
    Read optimized parameters as saved in a dict
    Path should not include the file type (like .json)
    """

    with open(path + ".json", "r") as json_file:
        best_params_dict = json.load(json_file)

    return best_params_dict


def get_ece(predicted_posterior, predicted_label, true_label, num_bins=40):
    """
    Return ece score
    Function borrowed from: https://github.com/neurodata/kdg/blob/main/kdg/utils.py
    """

    bin_size = 1 / num_bins
    total_sample = len(true_label)
    posteriors = predicted_posterior.max(axis=1)

    score = 0
    for bin in range(num_bins):
        indx = np.where(
            (posteriors > bin * bin_size) & (posteriors <= (bin + 1) * bin_size)
        )[0]

        acc = (
            np.nan_to_num(np.mean(predicted_label[indx] == true_label[indx]))
            if indx.size != 0
            else 0
        )
        conf = np.nan_to_num(np.mean(posteriors[indx])) if indx.size != 0 else 0
        score += len(indx) * np.abs(acc - conf)

    score /= total_sample
    return score


def find_indices_train_val_test(
    X_shape,
    ratio=[2, 1, 1],
    keys_types=["train", "val", "test"],
    dict_data_indices={},
    dataset_ind=0,
):
    ratio = np.array(ratio)
    ratio_base = np.linspace(0, X_shape, np.sum(ratio) + 1)
    ratio_base = ratio_base.astype(int)
    ratio_base_limits = [0] + ratio_base[np.cumsum(ratio)[:-1]].tolist()
    list_indices = np.arange(X_shape)
    np.random.shuffle(list_indices)
    dict_data_indices[dataset_ind] = {}
    for ind_counter, ind_min in enumerate(ratio_base_limits[:-1]):
        ind_max = ratio_base_limits[ind_counter + 1]
        cur_indices = list_indices[ind_min:ind_max]
        dict_data_indices[dataset_ind][keys_types[ind_counter]] = cur_indices
    
    return dict_data_indices