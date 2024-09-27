"""
Coauthors: Michael Ainsworth
           Jayanta Dey
           Haoyin Xu
           Noga Mudrik
"""

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score
import time
from toolbox import *

"""
Parameters
"""

num_classes = 10
reload_data = False  # indicator of whether to upload the data again

path_save = "metrics/cc18_all_parameters_test"
path_params = "metrics/dict_parameters_test"
path_train_val_test_indices = "metrics/dict_data_indices_test"
with open(path_params + ".json", "r") as json_file:
    dictionary_params = json.load(json_file)

with open(path_train_val_test_indices + ".json", "r") as json_file:
    dict_data_indices = json.load(json_file)

path = path_save  # "metrics/cc18_all_parameters"
type_file = ".json"
dataset_indices_max = dictionary_params["dataset_indices_max"]
max_shape_to_run = dictionary_params["max_shape_to_run"]
file_type_to_load = ".json"

"""
Number of repetitions for each CV fold at each sample size
"""

reps = 1
shape_2_all_sample_sizes = dictionary_params["shape_2_all_sample_sizes"]
shape_2_evolution = dictionary_params["shape_2_evolution"]
n_splits = dictionary_params["shape_2_evolution"]

models_to_run = {"RF": 1, "DN": 1, "GBDT": 1}


"""
Import CC18 data and pretuned hyperparameters
"""

if reload_data or "dataset_name" not in locals():
    X_data_list, y_data_list, dataset_name = load_cc18()

"""
Upload best parameters
"""

if "best_params_dict" not in locals():
    best_params_dict = open_data(path, "text_dict")

train_indices = [i for i in range(dataset_indices_max)]

"""
Create empty arrays to index sample sizes, kappa values, and ece scores
"""

all_sample_sizes = np.zeros((len(train_indices), shape_2_all_sample_sizes))

"""
Empty arrays to index times into
"""

train_times = {
    model_name: None for model_name in dictionary_params["classifiers_names"]
}
test_times = {model_name: None for model_name in dictionary_params["classifiers_names"]}

train_test_times = {model_name: {} for model_name in best_params_dict.keys()}


evolution_dict = {
    metric: {model_name: {} for model_name in best_params_dict.keys()}
    for metric in ["cohen_kappa", "ece"]
}

"""
For each dataset, train and predict for every sample size
Record outputs using Cohen's Kappa and ECE
"""

for dataset_index, dataset in enumerate(train_indices):
    print("\n\nCurrent Dataset: ", dataset)

    X = X_data_list[dataset]
    y = y_data_list[dataset]
    train_indices = dict_data_indices[dataset_index]["train"]
    test_indices = dict_data_indices[dataset_index]["test"]

    # If data set has over 10000 samples, resample to contain 10000
    if X.shape[0] > max_shape_to_run:
        X, y = sample_large_datasets(X, y)

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    training_sample_sizes = np.geomspace(
        len(np.unique(y_train)) * 5, X_train.shape[0], num=8, dtype=int
    )
    ss_inds = random_sample_new(
        X_train, y_train, training_sample_sizes, classes=num_classes
    )

    # Iterate through each sample size per dataset
    for sample_size_index, real_sample_size in enumerate(training_sample_sizes):
        cohen_ece_results_dict = {metric: {} for metric in ["cohen_kappa", "ece"]}
        train_test_times_cur = {
            model_name: np.zeros((reps)) for model_name in best_params_dict.keys()
        }

        X_train_new = X_train[ss_inds[sample_size_index]]
        y_train_new = y_train[ss_inds[sample_size_index]]

        # Repeat for number of repetitions, averaging results
        for model_name, model_best_params in best_params_dict.items():
            if models_to_run[model_name]:
                if dataset not in train_test_times["cohen_kappa"][model_name].keys():
                    train_test_times["cohen_kappa"][model_name][dataset] = {}
                    train_test_times["ece"][model_name][dataset] = {}

                try:
                    model = model_define(model_name, best_params_dict, dataset_index)

                    start_time = time.perf_counter()
                    model.fit(X_train_new, y_train_new)
                    end_time = time.perf_counter()
                    train_time = end_time - start_time

                    predictions = model.predict(X_test)
                    predict_probas = model.predict_proba(X_test)

                    cohen_kappa = cohen_kappa_score(y_test, predictions)
                    cohen_ece_results_dict["cohen_kappa"][model_name] = cohen_kappa
                    ece = get_ece(predict_probas, predictions, y_test)
                    cohen_ece_results_dict["ece"][model_name] = ece

                    train_test_times_cur[model_name] = train_time

                except ValueError:
                    print(model_name)

                if dataset not in evolution_dict["cohen_kappa"][model_name].keys():
                    evolution_dict["cohen_kappa"][model_name][dataset] = {}
                    evolution_dict["ece"][model_name][dataset] = {}
                if (
                    real_sample_size
                    not in evolution_dict["cohen_kappa"][model_name][dataset].keys()
                ):
                    evolution_dict["cohen_kappa"][model_name][dataset][
                        real_sample_size
                    ] = []
                    evolution_dict["ece"][model_name][dataset][real_sample_size] = []
                    train_test_times[model_name][dataset][real_sample_size] = []
                    train_test_times[model_name][dataset][real_sample_size] = []

                evolution_dict["cohen_kappa"][model_name][dataset][
                    real_sample_size
                ].append(cohen_ece_results_dict["cohen_kappa"][model_name])
                evolution_dict["ece"][model_name][dataset][real_sample_size].append(
                    cohen_ece_results_dict["ece"][model_name]
                )
                evolution_dict[model_name][dataset][real_sample_size].append(
                    train_test_times_cur[model_name]
                )

                # Changing the results to tuple enabling easier saving to json and ectacting the fata after that.

                train_test_times[model_name][dataset][real_sample_size] = tuple(
                    train_test_times[model_name][dataset][real_sample_size]
                )

                evolution_dict["cohen_kappa"][model_name][dataset][
                    real_sample_size
                ] = tuple(
                    evolution_dict["cohen_kappa"][model_name][dataset][real_sample_size]
                )

                evolution_dict["ece"][model_name][dataset][real_sample_size] = tuple(
                    evolution_dict["ece"][model_name][dataset][real_sample_size]
                )

    # Record sample sizes used
    all_sample_sizes[dataset_index][:] = np.array(training_sample_sizes)

new_dict = {}
for key_met in evolution_dict.keys():
    new_dict[key_met] = mod_dict(evolution_dict[key_met], tuple)
new_dict_times = mod_dict(train_test_times, tuple)


"""
Save sample sizes and model results in json files
"""


save_methods = {"json": 1}
save_methods_rewrite = {"json": 1}
save_best_parameters(
    save_methods,
    save_methods_rewrite,
    "metrics/cc18_sample_sizes_test.json",
    all_sample_sizes,
)

save_best_parameters(
    save_methods, save_methods_rewrite, "results/cc18_kappa_and_ece_test", new_dict
)
save_best_parameters(
    save_methods, save_methods_rewrite, "results/cc18_training_times_test", new_dict_times
)