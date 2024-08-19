from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score, accuracy_score
import time
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
import itertools
from toolbox import *

import warnings
warnings.filterwarnings("ignore")


# parameters 
dataset_indices_max = 7
max_shape_to_run = 10000
alpha_range_nn = [0.001, 0.01, 0.1]
subsample = [0.5, 0.8, 1.0]
num_classes = 10
dataset_indices = list(range(dataset_indices_max))
dict_data_indices = {dataset_ind: {} for dataset_ind in dataset_indices}
reload_data = False  # indicator of whether to upload the data again
dataset_indices = list(range(dataset_indices_max))
dict_data_indices = {dataset_ind: {} for dataset_ind in dataset_indices}
prefix = "new_results"

# Load data
SUITE_ID = [334]
X_data_list, y_data_list, dataset_name = import_datasets(SUITE_ID)

RF = 0
XGBT = 0
DN = 1

def load_params(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


path_rf = "SmacResults/RF_params.json"
path_xgbt = "SmacResults/XGBT_params.json"
path_tab = "SmacResults/Tab_params.json"

params_rf = load_params(path_rf)
params_xgbt = load_params(path_xgbt)
params_tab = load_params(path_tab)

# print(len(params_rf))
# print(len(params_xgbt))
# print(len(params_tab))

for dataset_index, dataset in enumerate(dataset_indices):
    print("\n\nCurrent Dataset: ", dataset)

    X = X_data_list[dataset]
    y = y_data_list[dataset]

    # If data set has over 10000 samples, resample to contain 10000
    if X.shape[0] > max_shape_to_run:
        X, y = sample_large_datasets(X, y)
    
    np.random.seed(dataset_index)
    dict_data_indices = find_indices_train_val_test(
        X.shape[0], dict_data_indices=dict_data_indices, dataset_ind=dataset_index
    )

    train_indices = dict_data_indices[dataset_index]["train"]
    val_indices = dict_data_indices[dataset_index]["val"]
    test_indices = dict_data_indices[dataset_index]["test"]

    ### Covert labels to numerical values
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    # y = pd.DataFrame(y_encoded)
    y = y_encoded

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    categorical_columns = X.select_dtypes(include=['object']).columns
    numeric_columns = X.select_dtypes(include=['number']).columns

    encoder = OneHotEncoder(sparse_output=False)
    if len(categorical_columns) > 0:
        X_encoded_strings = encoder.fit_transform(X[categorical_columns])

        X = np.hstack((X[numeric_columns].values, X_encoded_strings))
        print("Encoded", len(categorical_columns), " columns")
        encode_cnt += 1
    # else:
    #     print("No string columns to encode")

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    if RF == 1:
        print("\n Training Random Forest")

        ### For tuned parameters
        current_params = params_rf.get(dataset_name[dataset_index], {})
        model = RandomForestClassifier(random_state=317, **current_params)

        ### For default parameters
        # model = RandomForestClassifier(random_state=317)

        start = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        end_time = time.time()
        test_probs = model.predict_proba(X_val)

        train_time = end_time - start
        acc = accuracy_score(y_val, y_pred)
        kappa = cohen_kappa_score(y_val, y_pred)
        ece = get_ece(test_probs, y_pred, y_val)

        write_json(f"{prefix}/rf/{dataset_name[dataset_index]}/time_tuned.json", train_time)
        write_json(f"{prefix}/rf/{dataset_name[dataset_index]}/acc_tuned.json", acc)
        write_json(f"{prefix}/rf/{dataset_name[dataset_index]}/kappa_tuned.json", kappa)
        write_json(f"{prefix}/rf/{dataset_name[dataset_index]}/ece_tuned.json", ece)


        print("RF Time: ", train_time)
        print("RF Accuracy: ", acc)
        print("RF Cohen Kappa score: ", kappa)
        print("RF ECE score: ", ece)

    if XGBT == 1:
        print("\n Training XGBoost")

        ### For tuned parameters
        current_params = params_xgbt.get(dataset_name[dataset_index], {})
        model = xgb.XGBClassifier(random_state=317, **current_params)

        ### For default parameters
        # model = xgb.XGBClassifier(random_state=317)

        start = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        end_time = time.time()
        test_probs = model.predict_proba(X_val)

        train_time = end_time - start
        acc = accuracy_score(y_val, y_pred)
        kappa = cohen_kappa_score(y_val, y_pred)
        ece = get_ece(test_probs, y_pred, y_val)

        write_json(f"{prefix}/xgbt/{dataset_name[dataset_index]}/time_tuned.json", train_time)
        write_json(f"{prefix}/xgbt/{dataset_name[dataset_index]}/acc_tuned.json", acc)
        write_json(f"{prefix}/xgbt/{dataset_name[dataset_index]}/kappa_tuned.json", kappa)
        write_json(f"{prefix}/xgbt/{dataset_name[dataset_index]}/ece_tuned.json", ece)

        print("RF Time: ", train_time)
        print("RF Accuracy: ", acc)
        print("RF Cohen Kappa score: ", kappa)
        print("RF ECE score: ", ece )

    if DN == 1:
        print("\n Training TabNet")

        ### For tuned parameters
        current_params = params_tab.get(dataset_name[dataset_index], {})
        fit_params = {
            'max_epochs': current_params.pop('max_epochs', 100),  
            'batch_size': current_params.pop('batch_size', 1024)  
        }
        if current_params.get('solver') == 'adam':
            optimizer_fn = torch.optim.Adam
        elif current_params.get('solver') == 'sgd':
            optimizer_fn = torch.optim.SGD
        current_params.pop('solver', None)
        current_params['optimizer_fn'] = optimizer_fn
        optimizer_params = {'lr': current_params.pop('lr', 0.02)}
        current_params['optimizer_params'] = optimizer_params
        model = TabNetClassifier(seed=317, verbose=0, **current_params)
        start = time.time()
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], **fit_params)

        ### For default params
        # model = TabNetClassifier(seed=317, verbose=0)
        # start = time.time()
        # model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        y_pred = model.predict(X_val)
        end_time = time.time()
        test_probs = model.predict_proba(X_val)

        train_time = end_time - start
        acc = accuracy_score(y_val, y_pred)
        kappa = cohen_kappa_score(y_val, y_pred)
        ece = get_ece(test_probs, y_pred, y_val)

        write_json(f"{prefix}/tabnet/{dataset_name[dataset_index]}/time_tuned.json", train_time)
        write_json(f"{prefix}/tabnet/{dataset_name[dataset_index]}/acc_tuned.json", acc)
        write_json(f"{prefix}/tabnet/{dataset_name[dataset_index]}/kappa_tuned.json", kappa)
        write_json(f"{prefix}/tabnet/{dataset_name[dataset_index]}/ece_tuned.json", ece)

        print("RF Time: ", train_time)
        print("RF Accuracy: ", acc)
        print("RF Cohen Kappa score: ", kappa)
        print("RF ECE score: ", ece)











    break



