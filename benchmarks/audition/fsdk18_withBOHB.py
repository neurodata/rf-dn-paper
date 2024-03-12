"""
Coauthors: Haoyin Xu
           Yu-Chung Peng
           Madi Kusmanov
           Adway Kanhere
"""
from toolbox_ziyan import *
import argparse
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper
from scipy.optimize import fmin
import xgboost as xgb

import pandas as pd
import torchvision.models as models
import warnings
import random
import pickle

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from smac import Scenario
from pathlib import Path
from smac.facade import AbstractFacade
import matplotlib.pyplot as plt
from smac import MultiFidelityFacade as MFFacade
from smac.intensifier.hyperband import Hyperband
from smac.intensifier.successive_halving import SuccessiveHalving
import re


warnings.filterwarnings("ignore")

np.random.seed(317)

def run_GBT():
    gbt_kappa = []
    gbt_ece = []
    gbt_train_time = []
    gbt_test_time = []
    gbt_probs_labels = []
    storage_dict = {}
    #grid search
    #classes
    for classes in classes_space:
        # print(classes)
        d1 = {}
        # cohen_kappa vs num training samples (gbt)
        for samples in samples_space:
            l3 = []
            # train data
            # print("init GBT model")
            gbt_model = xgb.XGBClassifier(
                n_estimators=600,
                max_depth=32,
                learning_rate=0.1,
                objective='multi:softprob',
                random_state=317
            )
            # print("running gbt on sample:", samples)
            # lct = 0
            # print(type(fsdk18_test_images))
            # for img in fsdk18_test_images:
            #     image = img.reshape(32, 32)
            #     plt.imshow(image)
            #     label = index_to_labels.get(fsdk18_test_labels[lct])
            #     plt.title("{} | {}".format(fsdk18_test_labels[lct], label))
            #     lct += 1
            #     plt.show()
            cohen_kappa, ece, train_time, test_time, test_probs, test_labels, test_preds = run_gbt_image_set(
                gbt_model,
                fsdk18_train_images,
                fsdk18_train_labels,
                fsdk18_test_images,
                fsdk18_test_labels,
                samples,
                classes,
            )
            gbt_kappa.append(cohen_kappa)
            gbt_ece.append(ece)
            gbt_train_time.append(train_time)
            gbt_test_time.append(test_time)
            classes = sorted(classes)
            gbt_probs_labels.append("Classes:" + str(classes))
            gbt_probs_labels.append("Sample size:" + str(samples))
            for i in range(len(test_probs)):
                gbt_probs_labels.append("Posteriors:"+str(test_probs[i]) + ", " + "Test Labels:" + str(test_labels[i]))
            gbt_probs_labels.append(" \n")
            for i in range(len(test_probs)):
                l3.append([test_probs[i].tolist(), test_labels[i]])
            d1[samples] = l3
        storage_dict[tuple(sorted(classes))] = d1
    # print(gbt_probs_labels)
    # switch the classes and sample sizes
    switched_storage_dict = {}
    for classes, class_data in storage_dict.items():
        for samples, data in class_data.items():
            if samples not in switched_storage_dict:
                switched_storage_dict[samples] = {}
            if classes not in switched_storage_dict[samples]:
                switched_storage_dict[samples][classes] = data
    with open(prefix +'gbt_switched_storage_dict.pkl', 'wb') as f:
        pickle.dump(switched_storage_dict, f)
    # save the model
    with open(prefix + 'gbt_org.pkl', 'wb') as f:
        pickle.dump(gbt_model, f)
    print("gbt finished")
    write_result(prefix + "gbt_kappa_best.txt", gbt_kappa)
    write_result(prefix + "gbt_ece.txt", gbt_ece)
    write_result(prefix + "gbt_train_time.txt", gbt_train_time)
    write_result(prefix + "gbt_test_time.txt", gbt_test_time)
    write_result(prefix + "gbt_probs&labels.txt", gbt_probs_labels)
    write_json(prefix + "gbt_kappa_best.json", gbt_kappa)
    write_json(prefix + "gbt_ece.json", gbt_ece)
    write_json(prefix + "gbt_train_time.json", gbt_train_time)
    write_json(prefix + "gbt_test_time.json", gbt_test_time)

def run_BOHB():
    config_space = CS.ConfigurationSpace()
    config_space.add_hyperparameter(CS.UniformFloatHyperparameter('lr', lower=0.001, upper=0.1))
    solver = CSH.CategoricalHyperparameter('optimizer_name', ['sgd', 'adam'])
    config_space.add_hyperparameter(solver)
    batch_size = CSH.UniformFloatHyperparameter('batch_size', lower = 32, upper = 1024)
    config_space.add_hyperparameter(batch_size)
    epochs = CSH.UniformFloatHyperparameter('epochs', lower = 30, upper = 120)
    config_space.add_hyperparameter(epochs)

    trial_metrics = []
        
    for intensifier_object in [SuccessiveHalving, Hyperband]:
        # Define our environment variables
        scenario = Scenario(
            configspace=config_space,
            output_directory=Path("TEST_SMAC"),
            walltime_limit=30,  # Limit to two minutes
            n_trials=15000,  # Evaluated max 500 trials
            n_workers=8,  # Use one worker
            min_budget = 5,
            max_budget = 50,
        )

        # We want to run five random configurations before starting the optimization.
        initial_design = MFFacade.get_initial_design(scenario, n_configs=5)

        # Create our intensifier
        intensifier = intensifier_object(scenario, incumbent_selection="highest_budget")

        #seconds elapsed during training


        # The multi-fidelity facade is the closest implementation to BOHB. | https://automl.github.io/SMAC3/main/3_getting_started.html
        facades: list[AbstractFacade] = []
        smac = MFFacade(scenario=scenario, target_function=target_function_network)
        # smac = MFFacade(scenario=scenario, target_function=target_function(config, seed=317, budget=20))
        incumbent = smac.optimize()
        best_config = incumbent
        print(best_config)
        facades.append(smac)


def target_function_network(target_wrapper, config, seed, budget, train_data, train_labels, valid_data, valid_labels):
    print("INSIDE TARGET FUNC")
    start_time = time.time()
    """
    Trains the MLP with the given configuration and evaluates on the validation set.
    :param config: A Configuration (dictionary) of hyperparameters.
    :return: The validation error to be minimized.
    """
    # print(config)
    # Instantiate the MLPWrapper_forBOHB with the current configuration
    wrapper = target_wrapper(lr=config['lr'], 
                            optimizer_name=config['optimizer_name'], 
                            batch_size=config['batch_size'],
                            epochs=config['epochs'],
                            Valid_X = valid_data,
                            Valid_y = valid_labels,)

    # Fit the model on training data
    # time.sleep(1000)
    wrapper.fit(train_data, train_labels)

    predictions = wrapper.predict(valid_data)
    # quit(0)
    # print('OUTSIDE PRED')
    accuracy = accuracy_score(valid_labels, predictions)
    cohens_kappa = cohen_kappa_score(valid_labels, predictions)

    predicted_posterior = wrapper.predict_proba(valid_data)
    num_bins = np.unique(np.concatenate((train_labels, valid_labels))).size
    # print("HERE, HERE, HERE")
    
    # print(predicted_posterior)

    ece = get_ece(predicted_posterior, predictions, valid_labels, num_bins)

    # print('VALS CALCULATED')
    # Evaluate the model on validation data
    # The target function returns 1 - accuracy, so we directly return this value
    validation_error = wrapper.score(valid_data, valid_labels)
    
    end_time = time.time()  # End time for the trial
    trial_duration = end_time - start_time
    print('APPENDING')
    trial_data = {
        'config': str(config),
        'seed': seed,
        'budget': budget,
        'accuracy': accuracy,
        'cohens_kappa': cohens_kappa,
        'ece': ece,
        'trial_duration': trial_duration,
        'validation_error': validation_error
    }
    # print(trial_metrics)
    formatted_data = f"Trial Data:\n{json.dumps(trial_data, indent=4)}\n"
    with open('TEST_BOHB_FSDK18_cnn32.txt', 'a+') as f:
        f.write(formatted_data)
        f.write("\n" * 2)  # Add extra newlines for separation between entries

    print('APPENDED')

    return validation_error


def run_naive_rf():
    naive_rf_kappa = []
    naive_rf_ece = []
    naive_rf_train_time = []
    naive_rf_test_time = []
    navie_rf_probs_labels = []
    storage_dict = {}

    # RF = RandomForestClassifier(min_samples_split=2, min_samples_leaf=1, max_features=None ,n_jobs=-1, random_state=317)

    # ### Bayesian search for best parameters
    # time_limit = 7200 # 2 hours  
    # deadline_stopper = DeadlineStopper(time_limit)
    # start_time = time.perf_counter()
    # RF = RandomForestClassifier(min_samples_split=2, min_samples_leaf=1, max_features=None ,n_jobs=-1, random_state=317)

    # param_space = {
    #     'n_estimators': list(range(400, 1201, 100)), 
    #     'max_depth': list(range(2, 21, 2)),
    #     'min_samples_split': list(range(2, 11, 2)),
    #     'min_samples_leaf': list(range(1, 11, 2)),        
    # }

    # Bayes = BayesSearchCV(
    #     estimator=RF, 
    #     search_spaces=param_space,
    #     n_iter=50,  
    #     cv=3,  
    #     n_jobs=-1, 
    #     verbose=1, 
    # )

    # Bayes.fit(fsdk18_train_images, fsdk18_train_labels, callback=deadline_stopper)

    # best_params = Bayes.best_params_
    # end_time = time.perf_counter()
    # search_time = end_time - start_time
    # print("Best Accuracy:", Bayes.best_score_)
    # print("Best Hyperparameters:", best_params)
    # print("Bayesian Search time:", search_time)



    for classes in classes_space:
        d1 = {}
        # cohen_kappa vs num training samples (naive_rf)
        for samples in samples_space:
            l3 = []            
            # train data

            # RF_best = RandomForestClassifier(n_jobs=-1, random_state=317)

            # Best set of hyperparameters of 3 classes: 
            # RF_best = RandomForestClassifier(n_estimators=600, max_depth=16, min_samples_split=2, min_samples_leaf=1, max_features=None ,n_jobs=-1, random_state=317)

            # Best set of hyperparameters of 8 classes: 
            # RF_best = RandomForestClassifier(n_estimators=600, max_depth=32, min_samples_split=2, min_samples_leaf=1, max_features=None ,n_jobs=-1, random_state=317)

            # Best set of hyperparameters of 15 classes:
            RF_best = RandomForestClassifier(n_estimators=634, max_depth=35, min_samples_split=6, min_samples_leaf=1, max_features=None ,n_jobs=-1, random_state=317)

            cohen_kappa, ece, train_time, test_time, test_probs, test_labels, test_preds = run_rf_image_set(
                RF_best,
                fsdk18_train_images,
                fsdk18_train_labels,
                fsdk18_test_images,
                fsdk18_test_labels,
                samples,
                classes,
            )
            naive_rf_kappa.append(cohen_kappa)
            naive_rf_ece.append(ece)
            naive_rf_train_time.append(train_time)
            naive_rf_test_time.append(test_time)

            classes = sorted(classes)
            navie_rf_probs_labels.append("Classes:" + str(classes))

            navie_rf_probs_labels.append("Sample size:" + str(samples))

            for i in range(len(test_probs)):
                navie_rf_probs_labels.append("Posteriors:"+str(test_probs[i]) + ", " + "Test Labels:" + str(test_labels[i]))
            navie_rf_probs_labels.append(" \n")

            for i in range(len(test_probs)):
                l3.append([test_probs[i].tolist(), test_labels[i]])

            d1[samples] = l3

        storage_dict[tuple(sorted(classes))] = d1

    # switch the classes and sample sizes
    switched_storage_dict = {}

    for classes, class_data in storage_dict.items():

        for samples, data in class_data.items():

            if samples not in switched_storage_dict:
                switched_storage_dict[samples] = {}

            if classes not in switched_storage_dict[samples]:
                switched_storage_dict[samples][classes] = data

    with open(prefix +'rf_switched_storage_dict.pkl', 'wb') as f:
        pickle.dump(switched_storage_dict, f)

    # save the model
    with open(prefix + 'naive_rf_org.pkl', 'wb') as f:
        pickle.dump(RF_best, f)

    print("naive_rf finished")
    write_result(prefix + "naive_rf_kappa_best_smac.txt", naive_rf_kappa)
    write_result(prefix + "naive_rf_ece.txt", naive_rf_ece)
    write_result(prefix + "naive_rf_train_time.txt", naive_rf_train_time)
    write_result(prefix + "naive_rf_test_time.txt", naive_rf_test_time)
    write_result(prefix + "naive_rf_probs&labels.txt", navie_rf_probs_labels)
    write_json(prefix + "naive_rf_kappa_best.json", naive_rf_kappa)
    write_json(prefix + "naive_rf_ece.json", naive_rf_ece)
    write_json(prefix + "naive_rf_train_time.json", naive_rf_train_time)
    write_json(prefix + "naive_rf_test_time.json", naive_rf_test_time)


def run_cnn32():
    cnn32_kappa = []
    cnn32_ece = []
    cnn32_train_time = []
    cnn32_test_time = []
    cnn32_probs_labels = []
    storage_dict = {}

    # cnn32 = SimpleCNN32Filter(num_classes=18)

    # class CNN32Wrapper(BaseEstimator):
    #     def __init__(self, lr=0.01, batch_size=32, epochs=30, criterion = nn.CrossEntropyLoss(), optimizer_name='adam', Valid_X=None, Valid_y=None):
    #         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #         self.model = cnn32.to(self.device)
    #         self.lr = lr
    #         self.batch_size = batch_size
    #         self.epochs = epochs
    #         self.criterion = criterion
    #         self.optimizer_name = optimizer_name
    #         self.Valid_X = Valid_X
    #         self.Valid_y = Valid_y
    #         self.trial_start_time = time.perf_counter()
    #         self.trial_end_time = None

    #         if optimizer_name == 'sgd':
    #             self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
    #         elif optimizer_name == 'adam':
    #             self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    #         else:
    #             raise ValueError(f"Unknown optimizer: {optimizer_name}")

    #     def fit(self, X, y):
    #         max_epochs = [0]
    #         X = X.reshape(-1, 1, 32, 32)
    #         self.Valid_X = self.Valid_X.reshape(-1, 1, 32, 32)
    #         model = self.model
    #         criterion = self.criterion
    #         optimizer = self.optimizer
    #         prev_loss = float("inf")
    #         flag = 0
    #         for epoch in range(self.epochs):
    #             model.train()
    #             for i in range(0, len(X), self.batch_size):
    #                 inputs = X[i : i + self.batch_size].to(self.device)
    #                 labels = y[i : i + self.batch_size].to(self.device)
    #                 optimizer.zero_grad()
    #                 if inputs.shape[0] <= 2:
    #                     continue
    #                 outputs = model(inputs)
    #                 loss = criterion(outputs, labels)
    #                 loss.backward()
    #                 optimizer.step()

    #             model.eval()
    #             cur_loss = 0
    #             with torch.no_grad():
    #                 for i in range(0, len(self.Valid_X), self.batch_size):
    #                     # get the inputs
    #                     inputs = self.Valid_X[i : i + self.batch_size].to(self.device)
    #                     labels = self.Valid_y[i : i + self.batch_size].to(self.device)
    #                     if inputs.shape[0] == 1:
    #                         inputs = torch.cat((inputs, inputs, inputs), dim = 0)
    #                         labels = torch.cat((labels, labels, labels), dim = 0)

    #                     # forward
    #                     outputs = model(inputs)
    #                     loss = criterion(outputs, labels)
    #                     cur_loss += loss
    #             # early stop if 3 epochs in a row no loss decrease
    #             if cur_loss < prev_loss:
    #                 prev_loss = cur_loss
    #                 flag = 0
    #             else:
    #                 flag += 1
    #                 if flag >= 3:
    #                     max_epochs.append(epoch)
    #                     break
    #         return self
        
    #     def predict(self, X):
    #         X = X.reshape(-1, 1, 32, 32)
    #         model = self.model
    #         model.eval()
    #         test_preds = []
    #         with torch.no_grad():
    #             inputs = X.to(self.device)
    #             outputs = model(inputs)
    #             _, predicted = torch.max(outputs.data, 1)
    #             # predicted = predicted.unsqueeze(0)
    #             # test_preds = np.concatenate((test_preds, predicted.tolist()))
    #             return predicted.cpu().numpy()
            
    #     def predict_proba(self, X):
    #         X = X.reshape(-1, 1, 32, 32)
    #         model = self.model
    #         model.eval()
    #         test_probs = []
    #         with torch.no_grad():
    #             outputs = model(X.to(self.device))
    #             test_prob = nn.Softmax(dim=1)(outputs)
    #             test_probs = test_prob.cpu().numpy()
    #             return test_probs

    #     def score(self, X, y):
    #         X = X.reshape(-1, 1, 32, 32)
    #         model = self.model
    #         model.eval()
    #         predictions = self.predict(X)
    #         acc = accuracy_score(y, predictions)
    #         self.trial_end_time = time.perf_counter()
    #         valid_pred = self.predict(self.Valid_X)
    #         valid_acc = accuracy_score(self.Valid_y, valid_pred)
    #         valid_probs = self.predict_proba(self.Valid_X)
    #         valid_kappa = cohen_kappa_score(self.Valid_y, valid_pred)
    #         valid_ece = get_ece(valid_probs, valid_pred, self.Valid_y)
    #         with open('CNN32_BO_results_2h.txt', 'a') as f:
    #             f.write(f"Valid Accuracy: {valid_acc}, Valid Kappa: {valid_kappa}, Valid ECE: {valid_ece}, Time: {'placeholder'}\n")

    #         return acc

    # # train_images, train_labels, valid_images, valid_labels ,test_images, test_labels = prepare_data(fsdk18_train_images, fsdk18_train_labels, fsdk18_test_images, fsdk18_test_labels, samples_space[0] ,classes_space[0])
    # scaler = StandardScaler()
    # train_images = scaler.fit_transform(fsdk18_train_images)
    # valid_images = scaler.transform(fsdk18_valid_images)
    
    # train_images = torch.FloatTensor(train_images).unsqueeze(1)
    # train_labels = torch.LongTensor(fsdk18_train_labels)
    # valid_images = torch.FloatTensor(valid_images).unsqueeze(1)
    # valid_labels = torch.LongTensor(fsdk18_valid_labels)

    # # result = run_BOHB(cnn32, train_images, train_labels, valid_images, valid_labels)

    # # Bayesian optimization for best hyperparameters
    # time_limit = 240 # 2 hours  
    # deadline_stopper = DeadlineStopper(time_limit)
    # param_space={
    #     "batch_size": (32, 2048),
    #     "lr": (0.001, 0.1),
    #     "epochs": (30, 180),
    #     # "criterion": [nn.CrossEntropyLoss(), nn.NLLLoss()],
    #     "optimizer_name": ["adam", "sgd"],
    #     }

    # Bayes = BayesSearchCV(
    #     estimator=CNN32Wrapper(Valid_X=valid_images, Valid_y=valid_labels),
    #     search_spaces=param_space,
    #     n_iter=50,
    #     cv=3,
    #     verbose=1,
    #     n_jobs=-1,
    # )

    # with open('CNN32_BO_results_2h.txt', 'w') as f:
    #     pass

    # start_time = time.perf_counter()
    # Bayes.fit(train_images, train_labels, callback=deadline_stopper)
    # end_time = time.perf_counter()

    # iteration_times = deadline_stopper.iter_time
    # search_time = end_time - start_time
    # results = Bayes.cv_results_
    # results_df = pd.DataFrame(results)

    # best_params = Bayes.best_params_
    # print("Best Accuracy:", Bayes.best_score_)
    # print("Best Parameters:", best_params)
    # print("Bayesian Search Time:", search_time)

    # accuracy, kappa, ece, Time = [], [], [], []

    # newresults = []
    # with open('CNN32_BO_results_2h.txt', 'r') as f:
    #     data = f.readlines()
    #     linect = 0
    #     for line in data:
    #         if "Time:" in line:
    #             index = line.find("Time:")

    #             i = linect // 3 #+ linect % 3

    #             newtime = iteration_times[i]/3
    #             linect += 1
                
    #             newline = line[:index + len("Time: ")] + str(newtime) + "\n"
    #             newresults.append(newline)
    #         else:
    #             newresults.append(line)

    # with open('CNN32_BO_results_2h.txt', 'w') as f:
    #     f.writelines(newresults)

    # with open('CNN32_BO_results_2h.txt', 'r') as f:
    #     data = f.readlines()

    # for line in data:
    #     if "Time: " in line:
    #         numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)
    #         if numbers:
    #             accuracy.append(float(numbers[0]))
    #             kappa.append(float(numbers[1]))
    #             ece.append(float(numbers[2]))
    #             Time.append(float(numbers[3]))

    # print("sum time:", np.sum(Time)) 



    ### Best set of hyperparameters for 8 classes:
        # optimizer_name="adam",
        # epochs=100,
        # batch=1024,
        # lr=0.001,



    
    for classes in classes_space:
        d1 = {}

        # cohen_kappa vs num training samples (cnn32)
        for samples in samples_space:
            l3 = []
            # train data
            cnn32 = SimpleCNN32Filter(len(classes))
            # 3000 samples, 80% train is 2400 samples, 20% test
            train_images = trainx.copy()
            train_labels = trainy.copy()
            # reshape in 4d array
            test_images = testx.copy()
            test_labels = testy.copy()

            (
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
            ) = prepare_data(
                train_images, train_labels, test_images, test_labels, samples, classes
            )

            cohen_kappa, ece, train_time, test_time, test_probs, test_labels, test_preds = run_dn_image_es(
                cnn32,
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
                optimizer_name="adam",
                epochs=63,
                batch=881,
                lr=0.0011895644730894593,
                weight_decay=0.02881406216856244,
            )
            cnn32_kappa.append(cohen_kappa)
            cnn32_ece.append(ece)
            cnn32_train_time.append(train_time)
            cnn32_test_time.append(test_time)

            actual_test_labels = []
            for i in range(len(test_labels)):
                actual_test_labels.append(int(classes[test_labels[i]]))

            sorted_classes = sorted(classes)
            cnn32_probs_labels.append("Classes:" + str(sorted_classes))

            cnn32_probs_labels.append("Sample size:" + str(samples))
            
            actual_preds = []
            for i in range(len(test_preds)):
                actual_preds.append(int(sorted_classes[test_preds[i].astype(int)]))

            for i in range(len(test_probs)):
                cnn32_probs_labels.append("Posteriors:"+str(test_probs[i]) + ", " + "Test Labels:" + str(actual_test_labels[i]))
            cnn32_probs_labels.append(" \n")

            for i in range(len(test_probs)):
                l3.append([test_probs[i].tolist(), actual_test_labels[i]])

            d1[samples] = l3

        storage_dict[tuple(sorted(classes))] = d1

    # switch the classes and sample sizes
    switched_storage_dict = {}
    
    for classes, class_data in storage_dict.items():
        for samples, data in class_data.items():

            if samples not in switched_storage_dict:
                switched_storage_dict[samples] = {}

            if classes not in switched_storage_dict[samples]:
                switched_storage_dict[samples][classes] = data

    with open(prefix +'cnn32_switched_storage_dict.pkl', 'wb') as f:
        pickle.dump(switched_storage_dict, f)

    # save the model
    with open(prefix + 'cnn32.pkl', 'wb') as f:
        pickle.dump(cnn32, f)

    print("cnn32 finished")
    write_result(prefix + "cnn32_kappa_best_smac.txt", cnn32_kappa)
    write_result(prefix + "cnn32_ece_best.txt", cnn32_ece)
    write_result(prefix + "cnn32_train_time_best.txt", cnn32_train_time)
    write_result(prefix + "cnn32_test_time_best.txt", cnn32_test_time)
    write_result(prefix + "cnn32_probs&labels_best.txt", cnn32_probs_labels)
    write_json(prefix + "cnn32_kappa_best.json", cnn32_kappa)
    write_json(prefix + "cnn32_ece_best.json", cnn32_ece)
    write_json(prefix + "cnn32_train_time_best.json", cnn32_train_time)
    write_json(prefix + "cnn32_test_time_best.json", cnn32_test_time)


def run_cnn32_2l():
    cnn32_2l_kappa = []
    cnn32_2l_ece = []
    cnn32_2l_train_time = []
    cnn32_2l_test_time = []
    cnn32_2l_probs_labels = []
    storage_dict = {}

    # cnn32_2l = SimpleCNN32Filter2Layers(num_classes=18)

    # class CNN32Wrapper(BaseEstimator):
    #     def __init__(self, lr=0.01, batch_size=32, epochs=30, criterion = nn.CrossEntropyLoss(), optimizer_name='adam', Valid_X=None, Valid_y=None):
    #         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #         self.model = cnn32_2l.to(self.device)
    #         self.lr = lr
    #         self.batch_size = batch_size
    #         self.epochs = epochs
    #         self.criterion = criterion
    #         self.optimizer_name = optimizer_name
    #         self.Valid_X = Valid_X
    #         self.Valid_y = Valid_y
    #         self.trial_start_time = time.perf_counter()
    #         self.trial_end_time = None

    #         if optimizer_name == 'sgd':
    #             self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
    #         elif optimizer_name == 'adam':
    #             self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    #         else:
    #             raise ValueError(f"Unknown optimizer: {optimizer_name}")

    #     def fit(self, X, y):
    #         max_epochs = [0]
    #         X = X.reshape(-1, 1, 32, 32)
    #         self.Valid_X = self.Valid_X.reshape(-1, 1, 32, 32)
    #         model = self.model
    #         criterion = self.criterion
    #         optimizer = self.optimizer
    #         prev_loss = float("inf")
    #         flag = 0
    #         for epoch in range(self.epochs):
    #             model.train()
    #             for i in range(0, len(X), self.batch_size):
    #                 inputs = X[i : i + self.batch_size].to(self.device)
    #                 labels = y[i : i + self.batch_size].to(self.device)
    #                 optimizer.zero_grad()
    #                 if inputs.shape[0] <= 2:
    #                     continue
    #                 outputs = model(inputs)
    #                 loss = criterion(outputs, labels)
    #                 loss.backward()
    #                 optimizer.step()

    #             model.eval()
    #             cur_loss = 0
    #             with torch.no_grad():
    #                 for i in range(0, len(self.Valid_X), self.batch_size):
    #                     # get the inputs
    #                     inputs = self.Valid_X[i : i + self.batch_size].to(self.device)
    #                     labels = self.Valid_y[i : i + self.batch_size].to(self.device)
    #                     if inputs.shape[0] == 1:
    #                         inputs = torch.cat((inputs, inputs, inputs), dim = 0)
    #                         labels = torch.cat((labels, labels, labels), dim = 0)

    #                     # forward
    #                     outputs = model(inputs)
    #                     loss = criterion(outputs, labels)
    #                     cur_loss += loss
    #             # early stop if 3 epochs in a row no loss decrease
    #             if cur_loss < prev_loss:
    #                 prev_loss = cur_loss
    #                 flag = 0
    #             else:
    #                 flag += 1
    #                 if flag >= 3:
    #                     max_epochs.append(epoch)
    #                     break
    #         return self
        
    #     def predict(self, X):
    #         X = X.reshape(-1, 1, 32, 32)
    #         model = self.model
    #         model.eval()
    #         with torch.no_grad():
    #             inputs = X.to(self.device)
    #             outputs = model(inputs)
    #             _, predicted = torch.max(outputs.data, 1)
    #             return predicted.cpu().numpy()
            
    #     def predict_proba(self, X):
    #         X = X.reshape(-1, 1, 32, 32)
    #         model = self.model
    #         model.eval()
    #         test_probs = []
    #         with torch.no_grad():
    #             outputs = model(X.to(self.device))
    #             test_prob = nn.Softmax(dim=1)(outputs)
    #             test_probs = test_prob.cpu().numpy()
    #             return test_probs

    #     def score(self, X, y):
    #         X = X.reshape(-1, 1, 32, 32)
    #         model = self.model
    #         model.eval()
    #         predictions = self.predict(X)
    #         acc = accuracy_score(y, predictions)
    #         self.trial_end_time = time.perf_counter()
    #         valid_pred = self.predict(self.Valid_X)
    #         valid_acc = accuracy_score(self.Valid_y, valid_pred)
    #         valid_probs = self.predict_proba(self.Valid_X)
    #         valid_kappa = cohen_kappa_score(self.Valid_y, valid_pred)
    #         valid_ece = get_ece(valid_probs, valid_pred, self.Valid_y)
    #         with open('CNN32_2l_BO_results_2h.txt', 'a') as f:
    #             f.write(f"Valid Accuracy: {valid_acc}, Valid Kappa: {valid_kappa}, Valid ECE: {valid_ece}, Time: {'placeholder'}\n")

    #         return acc

    # # train_images, train_labels, valid_images, valid_labels ,test_images, test_labels = prepare_data(fsdk18_train_images, fsdk18_train_labels, fsdk18_test_images, fsdk18_test_labels, samples_space[0] ,classes_space[0])
    # scaler = StandardScaler()
    # train_images = scaler.fit_transform(fsdk18_train_images)
    # valid_images = scaler.transform(fsdk18_valid_images)
    
    # train_images = torch.FloatTensor(train_images).unsqueeze(1)
    # train_labels = torch.LongTensor(fsdk18_train_labels)
    # valid_images = torch.FloatTensor(valid_images).unsqueeze(1)
    # valid_labels = torch.LongTensor(fsdk18_valid_labels)

    # # result = run_BOHB(cnn32, train_images, train_labels, valid_images, valid_labels)

    # # Bayesian optimization for best hyperparameters
    # time_limit = 7200 # 2 hours  
    # deadline_stopper = DeadlineStopper(time_limit)
    # param_space={
    #     "batch_size": (32, 2048),
    #     "lr": (0.001, 0.1),
    #     "epochs": (30, 180),
    #     # "criterion": [nn.CrossEntropyLoss(), nn.NLLLoss()],
    #     "optimizer_name": ["adam", "sgd"],
    #     }

    # Bayes = BayesSearchCV(
    #     estimator=CNN32Wrapper(Valid_X=valid_images, Valid_y=valid_labels),
    #     search_spaces=param_space,
    #     n_iter=50,
    #     cv=3,
    #     verbose=1,
    #     n_jobs=-1,
    # )

    # with open('CNN32_2l_BO_results_2h.txt', 'w') as f:
    #     pass

    # start_time = time.perf_counter()
    # Bayes.fit(train_images, train_labels, callback=deadline_stopper)
    # end_time = time.perf_counter()

    # iteration_times = deadline_stopper.iter_time
    # search_time = end_time - start_time
    # results = Bayes.cv_results_
    # results_df = pd.DataFrame(results)

    # best_params = Bayes.best_params_
    # print("Best Accuracy:", Bayes.best_score_)
    # print("Best Parameters:", best_params)
    # print("Bayesian Search Time:", search_time)

    # accuracy, kappa, ece, Time = [], [], [], []

    # newresults = []
    # with open('CNN32_2l_BO_results_2h.txt', 'r') as f:
    #     data = f.readlines()
    #     linect = 0
    #     for line in data:
    #         if "Time:" in line:
    #             index = line.find("Time:")

    #             i = linect // 3 #+ linect % 3

    #             newtime = iteration_times[i]/3
    #             linect += 1
                
    #             newline = line[:index + len("Time: ")] + str(newtime) + "\n"
    #             newresults.append(newline)
    #         else:
    #             newresults.append(line)

    # with open('CNN32_2l_BO_results_2h.txt', 'w') as f:
    #     f.writelines(newresults)

    # with open('CNN32_2l_BO_results_2h.txt', 'r') as f:
    #     data = f.readlines()

    # for line in data:
    #     if "Time: " in line:
    #         numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)
    #         if numbers:
    #             accuracy.append(float(numbers[0]))
    #             kappa.append(float(numbers[1]))
    #             ece.append(float(numbers[2]))
    #             Time.append(float(numbers[3]))

    # print("sum time:", np.sum(iteration_times))
   






    for classes in classes_space:
        d1 = {}

        # cohen_kappa vs num training samples (cnn32_2l)
        for samples in samples_space:
            l3 = []
            # train data
            cnn32_2l = SimpleCNN32Filter2Layers(len(classes))
            # 3000 samples, 80% train is 2400 samples, 20% test
            train_images = trainx.copy()
            train_labels = trainy.copy()
            # reshape in 4d array
            test_images = testx.copy()
            test_labels = testy.copy()

            (
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
            ) = prepare_data(
                train_images, train_labels, test_images, test_labels, samples, classes
            )

            cohen_kappa, ece, train_time, test_time, test_probs, test_labels, test_preds = run_dn_image_es(
                cnn32_2l,
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
                optimizer_name="adam",
                batch=248,
                epochs=100,
                lr=0.001936799454289005,
                weight_decay=0.00010614767181282957,
            )
            cnn32_2l_kappa.append(cohen_kappa)
            cnn32_2l_ece.append(ece)
            cnn32_2l_train_time.append(train_time)
            cnn32_2l_test_time.append(test_time)

            actual_test_labels = []
            for i in range(len(test_labels)):
                actual_test_labels.append(int(classes[test_labels[i]]))

            sorted_classes = sorted(classes)
            cnn32_2l_probs_labels.append("Classes:" + str(classes))

            cnn32_2l_probs_labels.append("Sample size:" + str(samples))

            actual_preds = []
            for i in range(len(test_preds)):
                actual_preds.append(int(sorted_classes[test_preds[i].astype(int)]))
            
            for i in range(len(test_probs)):
                cnn32_2l_probs_labels.append("Posteriors:"+str(test_probs[i]) + ", " + "Test Labels:" + str(actual_test_labels[i]))
            cnn32_2l_probs_labels.append(" \n")

            for i in range(len(test_probs)):
                l3.append([test_probs[i].tolist(), actual_test_labels[i]])

            d1[samples] = l3

        storage_dict[tuple(sorted(classes))] = d1

    # switch the classes and sample sizes
    switched_storage_dict = {}

    for classes, class_data in storage_dict.items():
        for samples, data in class_data.items():

            if samples not in switched_storage_dict:
                switched_storage_dict[samples] = {}

            if classes not in switched_storage_dict[samples]:
                switched_storage_dict[samples][classes] = data

    with open(prefix + 'cnn32_2l_switched_storage_dict.pkl', 'wb') as f:
        pickle.dump(switched_storage_dict, f)

    # save the model
    with open(prefix + 'cnn32_2l.pkl', 'wb') as f:
        pickle.dump(cnn32_2l, f)

    print("cnn32_2l finished")
    write_result(prefix + "cnn32_2l_kappa_best_smac.txt", cnn32_2l_kappa)
    write_result(prefix + "cnn32_2l_ece.txt", cnn32_2l_ece)
    write_result(prefix + "cnn32_2l_train_time.txt", cnn32_2l_train_time)
    write_result(prefix + "cnn32_2l_test_time.txt", cnn32_2l_test_time)
    write_result(prefix + "cnn32_2l_probs&labels.txt", cnn32_2l_probs_labels)
    write_json(prefix + "cnn32_2l_kappa_best.json", cnn32_2l_kappa)
    write_json(prefix + "cnn32_2l_ece.json", cnn32_2l_ece)
    write_json(prefix + "cnn32_2l_train_time.json", cnn32_2l_train_time)
    write_json(prefix + "cnn32_2l_test_time.json", cnn32_2l_test_time)


def run_cnn32_5l():
    cnn32_5l_kappa = []
    cnn32_5l_ece = []
    cnn32_5l_train_time = []
    cnn32_5l_test_time = []
    cnn32_5l_probs_labels = []
    storage_dict = {}


    # # Grid search for best hyperparameters

    # cnn32_5l = SimpleCNN32Filter5Layers(num_classes=18)

    # class CNN32Wrapper(BaseEstimator):
    #     def __init__(self, lr=0.01, batch_size=32, epochs=30, criterion = nn.CrossEntropyLoss(), optimizer_name='adam', Valid_X=None, Valid_y=None):
    #         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #         self.model = cnn32_5l.to(self.device)
    #         self.lr = lr
    #         self.batch_size = batch_size
    #         self.epochs = epochs
    #         self.criterion = criterion
    #         self.optimizer_name = optimizer_name
    #         self.Valid_X = Valid_X
    #         self.Valid_y = Valid_y
    #         self.trial_start_time = time.perf_counter()
    #         self.trial_end_time = None

    #         if optimizer_name == 'sgd':
    #             self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
    #         elif optimizer_name == 'adam':
    #             self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    #         else:
    #             raise ValueError(f"Unknown optimizer: {optimizer_name}")

    #     def fit(self, X, y):
    #         max_epochs = [0]
    #         X = X.reshape(-1, 1, 32, 32)
    #         self.Valid_X = self.Valid_X.reshape(-1, 1, 32, 32)
    #         model = self.model
    #         criterion = self.criterion
    #         optimizer = self.optimizer
    #         prev_loss = float("inf")
    #         flag = 0
    #         for epoch in range(self.epochs):
    #             model.train()
    #             for i in range(0, len(X), self.batch_size):
    #                 inputs = X[i : i + self.batch_size].to(self.device)
    #                 labels = y[i : i + self.batch_size].to(self.device)
    #                 optimizer.zero_grad()
    #                 if inputs.shape[0] <= 2:
    #                     continue
    #                 outputs = model(inputs)
    #                 loss = criterion(outputs, labels)
    #                 loss.backward()
    #                 optimizer.step()

    #             model.eval()
    #             cur_loss = 0
    #             with torch.no_grad():
    #                 for i in range(0, len(self.Valid_X), self.batch_size):
    #                     # get the inputs
    #                     inputs = self.Valid_X[i : i + self.batch_size].to(self.device)
    #                     labels = self.Valid_y[i : i + self.batch_size].to(self.device)
    #                     if inputs.shape[0] == 1:
    #                         inputs = torch.cat((inputs, inputs, inputs), dim = 0)
    #                         labels = torch.cat((labels, labels, labels), dim = 0)

    #                     # forward
    #                     outputs = model(inputs)
    #                     loss = criterion(outputs, labels)
    #                     cur_loss += loss
    #             # early stop if 3 epochs in a row no loss decrease
    #             if cur_loss < prev_loss:
    #                 prev_loss = cur_loss
    #                 flag = 0
    #             else:
    #                 flag += 1
    #                 if flag >= 3:
    #                     max_epochs.append(epoch)
    #                     break
    #         return self
        
    #     def predict(self, X):
    #         X = X.reshape(-1, 1, 32, 32)
    #         model = self.model
    #         model.eval()
    #         test_preds = []
    #         with torch.no_grad():
    #             inputs = X.to(self.device)
    #             outputs = model(inputs)
    #             _, predicted = torch.max(outputs.data, 1)
    #             # predicted = predicted.unsqueeze(0)
    #             # test_preds = np.concatenate((test_preds, predicted.tolist()))
    #             return predicted.cpu().numpy()
            
    #     def predict_proba(self, X):
    #         X = X.reshape(-1, 1, 32, 32)
    #         model = self.model
    #         model.eval()
    #         test_probs = []
    #         with torch.no_grad():
    #             outputs = model(X.to(self.device))
    #             test_prob = nn.Softmax(dim=1)(outputs)
    #             test_probs = test_prob.cpu().numpy()
    #             return test_probs

    #     def score(self, X, y):
    #         X = X.reshape(-1, 1, 32, 32)
    #         model = self.model
    #         model.eval()
    #         predictions = self.predict(X)
    #         acc = accuracy_score(y, predictions)
    #         self.trial_end_time = time.perf_counter()
    #         valid_pred = self.predict(self.Valid_X)
    #         valid_acc = accuracy_score(self.Valid_y, valid_pred)
    #         valid_probs = self.predict_proba(self.Valid_X)
    #         valid_kappa = cohen_kappa_score(self.Valid_y, valid_pred)
    #         valid_ece = get_ece(valid_probs, valid_pred, self.Valid_y)
    #         with open('CNN32_5l_BO_results_2h.txt', 'a') as f:
    #             f.write(f"Valid Accuracy: {valid_acc}, Valid Kappa: {valid_kappa}, Valid ECE: {valid_ece}, Time: {'placeholder'}\n")

    #         return acc

    # # train_images, train_labels, valid_images, valid_labels ,test_images, test_labels = prepare_data(fsdk18_train_images, fsdk18_train_labels, fsdk18_test_images, fsdk18_test_labels, samples_space[0] ,classes_space[0])
    # scaler = StandardScaler()
    # train_images = scaler.fit_transform(fsdk18_train_images)
    # valid_images = scaler.transform(fsdk18_valid_images)
    
    # train_images = torch.FloatTensor(train_images).unsqueeze(1)
    # train_labels = torch.LongTensor(fsdk18_train_labels)
    # valid_images = torch.FloatTensor(valid_images).unsqueeze(1)
    # valid_labels = torch.LongTensor(fsdk18_valid_labels)

    # # result = run_BOHB(cnn32, train_images, train_labels, valid_images, valid_labels)

    # # Bayesian optimization for best hyperparameters
    # time_limit = 7200 # 2 hours  
    # deadline_stopper = DeadlineStopper(time_limit)
    # param_space={
    #     "batch_size": (32, 2048),
    #     "lr": (0.001, 0.1),
    #     "epochs": (30, 180),
    #     # "criterion": [nn.CrossEntropyLoss(), nn.NLLLoss()],
    #     "optimizer_name": ["adam", "sgd"],
    #     }

    # Bayes = BayesSearchCV(
    #     estimator=CNN32Wrapper(Valid_X=valid_images, Valid_y=valid_labels),
    #     search_spaces=param_space,
    #     n_iter=50,
    #     cv=3,
    #     verbose=1,
    #     n_jobs=-1,
    # )

    # with open('CNN32_5l_BO_results_2h.txt', 'w') as f:
    #     pass

    # start_time = time.perf_counter()
    # Bayes.fit(train_images, train_labels, callback=deadline_stopper)
    # end_time = time.perf_counter()

    # iteration_times = deadline_stopper.iter_time
    # search_time = end_time - start_time
    # results = Bayes.cv_results_
    # results_df = pd.DataFrame(results)

    # best_params = Bayes.best_params_
    # print("Best Accuracy:", Bayes.best_score_)
    # print("Best Parameters:", best_params)
    # print("Bayesian Search Time:", search_time)

    # accuracy, kappa, ece, Time = [], [], [], []

    # newresults = []
    # with open('CNN32_5l_BO_results_2h.txt', 'r') as f:
    #     data = f.readlines()
    #     linect = 0
    #     for line in data:
    #         if "Time:" in line:
    #             index = line.find("Time:")

    #             i = linect // 3 #+ linect % 3

    #             newtime = iteration_times[i]/3
    #             linect += 1
                
    #             newline = line[:index + len("Time: ")] + str(newtime) + "\n"
    #             newresults.append(newline)
    #         else:
    #             newresults.append(line)

    # with open('CNN32_5l_BO_results_2h.txt', 'w') as f:
    #     f.writelines(newresults)

    # with open('CNN32_5l_BO_results_2h.txt', 'r') as f:
    #     data = f.readlines()

    # for line in data:
    #     if "Time: " in line:
    #         numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)
    #         if numbers:
    #             accuracy.append(float(numbers[0]))
    #             kappa.append(float(numbers[1]))
    #             ece.append(float(numbers[2]))
    #             Time.append(float(numbers[3]))

    # print("sum time:", np.sum(iteration_times))




    for classes in classes_space:
        d1 = {}

        # cohen_kappa vs num training samples (cnn32_5l)
        for samples in samples_space:
            l3 = []
            # train data
            cnn32_5l = SimpleCNN32Filter5Layers(len(classes))
            # 3000 samples, 80% train is 2400 samples, 20% test
            train_images = trainx.copy()
            train_labels = trainy.copy()
            # reshape in 4d array
            test_images = testx.copy()
            test_labels = testy.copy()

            (
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
            ) = prepare_data(
                train_images, train_labels, test_images, test_labels, samples, classes
            )

            cohen_kappa, ece, train_time, test_time, test_probs, test_labels, test_preds = run_dn_image_es(
                cnn32_5l,
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
                optimizer_name="sgd",
                lr=0.001,
                epochs=100,
                batch=227,
                dampening=0.8295654584238159,
                momentum=0.9666948703632617,

                weight_decay=0.002546558192892438,
            )
            cnn32_5l_kappa.append(cohen_kappa)
            cnn32_5l_ece.append(ece)
            cnn32_5l_train_time.append(train_time)
            cnn32_5l_test_time.append(test_time)

            actual_test_labels = []
            for i in range(len(test_labels)):
                actual_test_labels.append(int(classes[test_labels[i]]))

            sorted_classes = sorted(classes)
            cnn32_5l_probs_labels.append("Classes:" + str(classes))

            cnn32_5l_probs_labels.append("Sample size:" + str(samples))

            actual_preds = []
            for i in range(len(test_preds)):
                actual_preds.append(int(sorted_classes[test_preds[i].astype(int)]))

            for i in range(len(test_probs)):
                cnn32_5l_probs_labels.append("Posteriors:"+str(test_probs[i]) + ", " + "Test Labels:" + str(actual_test_labels[i]))
            cnn32_5l_probs_labels.append(" \n")

            for i in range(len(test_probs)):
                l3.append([test_probs[i].tolist(), actual_test_labels[i]])

            d1[samples] = l3

        storage_dict[tuple(sorted(classes))] = d1

    # switch the classes and sample sizes
    switched_storage_dict = {}

    for classes, class_data in storage_dict.items():
        for samples, data in class_data.items():

            if samples not in switched_storage_dict:
                switched_storage_dict[samples] = {}

            if classes not in switched_storage_dict[samples]:
                switched_storage_dict[samples][classes] = data

    with open(prefix + 'cnn32_5l_switched_storage_dict.pkl', 'wb') as f:
        pickle.dump(switched_storage_dict, f)

    # save the model
    with open(prefix + 'cnn32_5l.pkl', 'wb') as f:
        pickle.dump(cnn32_5l, f)

    print("cnn32_5l finished")
    write_result(prefix + "cnn32_5l_kappa_best_smac.txt", cnn32_5l_kappa)
    write_result(prefix + "cnn32_5l_ece.txt", cnn32_5l_ece)
    write_result(prefix + "cnn32_5l_train_time.txt", cnn32_5l_train_time)
    write_result(prefix + "cnn32_5l_test_time.txt", cnn32_5l_test_time)
    write_result(prefix + "cnn32_5l_probs&labels.txt", cnn32_5l_probs_labels)
    write_json(prefix + "cnn32_5l_kappa.json", cnn32_5l_kappa)
    write_json(prefix + "cnn32_5l_ece.json", cnn32_5l_ece)
    write_json(prefix + "cnn32_5l_train_time.json", cnn32_5l_train_time)
    write_json(prefix + "cnn32_5l_test_time.json", cnn32_5l_test_time)

def train_resnet_wrapper(config, budget, X_train, y_train, X_valid, y_valid):
    """
    Train the ResnetWrapper model with given configuration and return the loss.

    Args:
        config (dict): Configuration dictionary containing hyperparameters.
        budget (float): Budget parameter from BOAH, not used in this context.
        X_train (ndarray): Training data.
        y_train (ndarray): Training labels.
        X_valid (ndarray): Validation data.
        y_valid (ndarray): Validation labels.

    Returns:
        dict: A dictionary containing the 'loss' and other information.
    """
    # Create an instance of ResnetWrapper with the provided config
    model = ResnetWrapper(**config, Valid_X=X_valid, Valid_y=y_valid)
    
    # Fit the model using the training data
    model.fit(X_train, y_train)

    # Here, you should implement the logic to evaluate the model.
    # For instance, you can use the model.score() method if you have it,
    # or any other metric like validation loss, accuracy, etc.
    # Assuming model.score() returns the accuracy, and you want to minimize the loss:
    validation_loss = 1 - model.score(X_valid, y_valid)  # Example metric

    # Return the result in the required format for BOAH optimization
    return {'loss': validation_loss, 'info': {'budget': budget}}

def run_resnet18():
    resnet18_kappa = []
    resnet18_ece = []
    resnet18_train_time = []
    resnet18_test_time = []
    resnet18_probs_labels = []
    storage_dict = {}



    # # Grid search for best hyperparameters

    # resnet = models.resnet18(pretrained=True)

    # num_ftrs = resnet.fc.in_features
    # resnet.fc = nn.Linear(num_ftrs, 18)

    # class ResnetWrapper(BaseEstimator):
    #     def __init__(self, lr=0.01, batch_size=32, epochs=30, criterion = nn.CrossEntropyLoss(), optimizer_name='adam', Valid_X=None, Valid_y=None):
    #         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #         self.model = resnet.to(self.device)
    #         self.lr = lr
    #         self.batch_size = batch_size
    #         self.epochs = epochs
    #         self.criterion = criterion
    #         self.optimizer_name = optimizer_name
    #         self.Valid_X = Valid_X
    #         self.Valid_y = Valid_y
    #         # self.momentum = momentum
    #         # self.weight_decay = weight_decay

    #         if optimizer_name == 'sgd':
    #             self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
    #         elif optimizer_name == 'adam':
    #             self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    #         else:
    #             raise ValueError(f"Unknown optimizer: {optimizer_name}")

    #     def fit(self, X, y):
    #         max_epochs = [0]
    #         X = X.reshape(-1, 3, 32, 32)
    #         self.Valid_X = self.Valid_X.reshape(-1, 3, 32, 32)
    #         model = self.model
    #         criterion = self.criterion
    #         optimizer = self.optimizer
    #         prev_loss = float("inf")
    #         flag = 0
    #         for epoch in range(self.epochs):
    #             model.train()
    #             for i in range(0, len(X), self.batch_size):
    #                 inputs = X[i : i + self.batch_size].to(self.device)
    #                 labels = y[i : i + self.batch_size].to(self.device)
    #                 optimizer.zero_grad()
    #                 if inputs.shape[0] <= 2:
    #                     continue
    #                 outputs = model(inputs)
    #                 loss = criterion(outputs, labels)
    #                 loss.backward()
    #                 optimizer.step()

    #             model.eval()
    #             cur_loss = 0
    #             with torch.no_grad():
    #                 for i in range(0, len(self.Valid_X), self.batch_size):
    #                     # get the inputs
    #                     inputs = self.Valid_X[i : i + self.batch_size].to(self.device)
    #                     labels = self.Valid_y[i : i + self.batch_size].to(self.device)
    #                     if inputs.shape[0] == 1:
    #                         inputs = torch.cat((inputs, inputs, inputs), dim = 0)
    #                         labels = torch.cat((labels, labels, labels), dim = 0)

    #                     # forward
    #                     outputs = model(inputs)
    #                     loss = criterion(outputs, labels)
    #                     cur_loss += loss
    #             # early stop if 3 epochs in a row no loss decrease
    #             if cur_loss < prev_loss:
    #                 prev_loss = cur_loss
    #                 flag = 0
    #             else:
    #                 flag += 1
    #                 if flag >= 3:
    #                     max_epochs.append(epoch)
    #                     break
    #                 else:
    #                     max_epochs.append(self.epochs)
    #         # print(np.max(max_epochs))
    #         return self
        
    #     def predict(self, X):
    #         X = X.reshape(-1, 3, 32, 32)
    #         model = self.model
    #         model.eval()
    #         with torch.no_grad():
    #             outputs = model(X.to(self.device))
    #             _, predicted = torch.max(outputs.data, 1)
    #             return predicted.cpu()

    #     def score(self, X, y):
    #         X = X.reshape(-1, 3, 32, 32)
    #         model = self.model
    #         model.eval()
    #         predictions = self.predict(X)
    #         acc = accuracy_score(y, predictions)
    #         return acc

    # # train_images, train_labels, valid_images, valid_labels ,test_images, test_labels = prepare_data(fsdk18_train_images, fsdk18_train_labels, fsdk18_test_images, fsdk18_test_labels, samples_space[0] ,classes_space[0])
    # scaler = StandardScaler()
    # train_images = scaler.fit_transform(fsdk18_train_images)
    # valid_images = scaler.transform(fsdk18_valid_images)
    
    # train_images = torch.FloatTensor(train_images).unsqueeze(1)
    # train_labels = torch.LongTensor(fsdk18_train_labels)
    # valid_images = torch.FloatTensor(valid_images).unsqueeze(1)
    # valid_labels = torch.LongTensor(fsdk18_valid_labels)

    # train_images = torch.cat((train_images, train_images, train_images), dim=1)
    # valid_images = torch.cat((valid_images, valid_images, valid_images), dim=1)

    # # Bayesian optimization for best hyperparameters
    # time_limit = 7200 # 2 hours  
    # deadline_stopper = DeadlineStopper(time_limit)
    # start_time = time.perf_counter()
    # param_space={
    #     "batch_size": [32, 64, 128 ,256, 512, 1024, 2048],
    #     "lr": [0.001, 0.01, 0.1],
    #     "epochs": list(range(60, 121, 10)),
    #     # "momentum": [0.7, 0.8, 0.9, 0.95, 0.99],
    #     # "weight_decay": [1e-4, 1e-5, 1e-6],
    #     # "criterion": [nn.CrossEntropyLoss(), nn.NLLLoss()],
    #     "optimizer_name": ["adam", "sgd"],
    #     }

    # Bayes = BayesSearchCV(
    #     estimator=ResnetWrapper(Valid_X=valid_images, Valid_y=valid_labels),
    #     search_spaces=param_space,
    #     n_iter=50,
    #     cv=3,
    #     verbose=1,
    #     n_jobs=-1,
    # )

    # Bayes.fit(train_images, train_labels, callback=deadline_stopper)

    # best_params = Bayes.best_params_
    # end_time = time.perf_counter()
    # search_time = end_time - start_time
    # print("Best Accuracy:", Bayes.best_score_)
    # print("Best Parameters:", best_params)
    # print("Bayesian Search Time:", search_time)
    # with open("Bayesian Search time res.txt", "w") as f:
    #     f.write(str(search_time)) 


    ### Best set of hyperparameters for 3 classes:L
        # epochs=60,
        # lr=0.001,
        # batch=32,
        # optimizer_name="adam",




    for classes in classes_space:
        d1 = {}

        # cohen_kappa vs num training samples (resnet18)
        for samples in samples_space:
            l3 = []
            resnet = models.resnet18(pretrained=True)

            num_ftrs = resnet.fc.in_features
            resnet.fc = nn.Linear(num_ftrs, len(classes))
            # train data
            # 3000 samples, 80% train is 2400 samples, 20% test
            train_images = trainx.copy()
            train_labels = trainy.copy()
            # reshape in 4d array
            test_images = testx.copy()
            test_labels = testy.copy()

            (
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
            ) = prepare_data(
                train_images, train_labels, test_images, test_labels, samples, classes
            )

            # need to duplicate channel because batch norm cant have 1 channel images
            train_images = torch.cat((train_images, train_images, train_images), dim=1)
            valid_images = torch.cat((valid_images, valid_images, valid_images), dim=1)
            test_images = torch.cat((test_images, test_images, test_images), dim=1)

            cohen_kappa, ece, train_time, test_time, test_probs, test_labels, test_preds = run_dn_image_es(
                resnet,
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
                epochs=87,
                lr=0.001,
                batch=32,
                optimizer_name="adam",
            )
            resnet18_kappa.append(cohen_kappa)
            resnet18_ece.append(ece)
            resnet18_train_time.append(train_time)
            resnet18_test_time.append(test_time)

            actual_test_labels = []
            for i in range(len(test_labels)):
                actual_test_labels.append(int(classes[test_labels[i]]))

            sorted_classes = sorted(classes)
            resnet18_probs_labels.append("Classes:" + str(classes))

            resnet18_probs_labels.append("Sample size:" + str(samples))

            actual_preds = []
            for i in range(len(test_preds)):
                actual_preds.append(int(sorted_classes[test_preds[i].astype(int)]))

            for i in range(len(test_probs)):
                resnet18_probs_labels.append("Posteriors:"+str(test_probs[i]) + ", " + "Test Labels:" + str(actual_test_labels[i]))
            resnet18_probs_labels.append(" \n")

            for i in range(len(test_probs)):
                l3.append([test_probs[i].tolist(), actual_test_labels[i]])
        
            d1[samples] = l3
        storage_dict[tuple(sorted(classes))] = d1

    # switch the classes and sample sizes
    switched_storage_dict = {}

    for classes, class_data in storage_dict.items():
        for samples, data in class_data.items():

            if samples not in switched_storage_dict:
                switched_storage_dict[samples] = {}

            if classes not in switched_storage_dict[samples]:
                switched_storage_dict[samples][classes] = data

    with open(prefix + 'resnet18_switched_storage_dict.pkl', 'wb') as f:
        pickle.dump(switched_storage_dict, f)

    # save the model
    with open(prefix + 'resnet18.pkl', 'wb') as f:
        pickle.dump(resnet, f)


    print("resnet18 finished")
    write_result(prefix + "resnet18_kappa_best_smac.txt", resnet18_kappa)
    write_result(prefix + "resnet18_ece.txt", resnet18_ece)
    write_result(prefix + "resnet18_train_time.txt", resnet18_train_time)
    write_result(prefix + "resnet18_test_time.txt", resnet18_test_time)
    write_result(prefix + "resnet18_probs&labels.txt", resnet18_probs_labels)
    write_json(prefix + "resnet18_kappa.json", resnet18_kappa)
    write_json(prefix + "resnet18_ece.json", resnet18_ece)
    write_json(prefix + "resnet18_train_time.json", resnet18_train_time)
    write_json(prefix + "resnet18_test_time.json", resnet18_test_time)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", help="class number")
    parser.add_argument("-f", help="feature type")
    parser.add_argument("-data", help="audio files location")
    parser.add_argument("-labels", help="labels file location")
    args = parser.parse_args()
    n_classes = int(args.m)
    feature_type = str(args.f)

    train_folder = str(args.data)
    train_label = pd.read_csv(str(args.labels))

    # select subset of data that only contains 300 samples per class
    labels_chosen = train_label[
        train_label["label"].map(train_label["label"].value_counts() >= 0)
    ]

    # # get the sample size of each class
    # sample_size = labels_chosen.label.value_counts().to_dict()
    # print("sample_sizes:", sample_size)
    # print("labels_chosen:", labels_chosen.iloc[:, 1:2].value_counts())

    training_files = []
    for file in os.listdir(train_folder):
        for x in labels_chosen.fname.to_list():
            if file.endswith(x):
                training_files.append(file)

    path_recordings = []
    for audiofile in training_files:
        path_recordings.append(os.path.join(train_folder, audiofile))

    # convert selected label names to integers
    labels_to_index = {
        "Acoustic_guitar": 0,
        "Applause": 1,
        "Bass_drum": 2,
        "Cello": 3,
        "Clarinet": 4,
        "Double_bass": 5,
        "Fart": 6,
        "Fireworks": 7,
        "Flute": 8,
        "Hi-hat": 9,
        "Laughter": 10,
        "Saxophone": 11,
        "Shatter": 12,
        "Snare_drum": 13,
        "Squeak": 14,
        "Tearing": 15,
        "Trumpet": 16,
        "Violin_or_fiddle": 17,
    }

    # encode labels to integers
    get_labels = labels_chosen["label"].replace(labels_to_index).to_list()
    labels_chosen = labels_chosen.reset_index()

    # data is normalized upon loading
    # load dataset
    x_spec, y_number = load_fsdk18(
        path_recordings, labels_chosen, get_labels, feature_type
    )

    nums = list(range(18))
    samples_space = np.geomspace(10, 450, num=6, dtype=int)
    # define path, samples space and number of class combinations
    if feature_type == "melspectrogram":
        prefix = args.m + "_class_mel/"
    elif feature_type == "spectrogram":
        prefix = args.m + "_class/"
    elif feature_type == "mfcc":
        prefix = args.m + "_class_mfcc/"

    # create list of classes with const random seed
    random.Random(5).shuffle(nums)
    classes_space = list(combinations_45(nums, n_classes))

    # scale the data
    # x_spec = x_spec[:5400] #reshape x_spec by Ziyan for testing, orginial shape was (11073, 32, 32)
    # print(x_spec.shape)
    x_spec = scale(x_spec.reshape(len(x_spec), -1), axis=1).reshape(len(x_spec), 32, 32)
    y_number = np.array(y_number)
    # y_number = y_number[:5400] #reshape x_spec by Ziyan for testing, orginial shape was (11073, 32, 32)

    # need to take train/valid/test equally from each class
    trainx, remainx, trainy, remainy = train_test_split(
        x_spec,
        y_number,
        shuffle=True,
        test_size=0.5,
        stratify=y_number,
    )

    testx, valx, testy, valy = train_test_split(
        remainx,
        remainy,
        shuffle=True,
        test_size=0.5,
        stratify=remainy,
    )

    # 3000 samples, 80% train is 2400 samples, 20% test
    fsdk18_train_images = trainx.reshape(-1, 32 * 32)
    fsdk18_train_labels = trainy.copy()
    # reshape in 2d array
    fsdk18_test_images = testx.reshape(-1, 32 * 32)
    fsdk18_test_labels = testy.copy()
    # validation set
    fsdk18_valid_images = valx.reshape(-1, 32 * 32)
    fsdk18_valid_labels = valy.copy()


    # print("Running GBT tuning \n")
    # run_GBT()

    print("Running RF tuning \n")
    run_naive_rf()

    # print("Running CNN32 tuning \n")
    # run_cnn32()

    # print("Running CNN32_2l tuning \n")
    # run_cnn32_2l()

    # print("Running CNN32_5l tuning \n")
    # run_cnn32_5l()

    # print("Running Resnet tuning \n")
    # run_resnet18()
    







