"""
Coauthors: Haoyin Xu
           Yu-Chung Peng
"""
from toolbox import *

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
import warnings
import random
import pickle

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

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

def run_GBT():
    gbt_acc = []
    gbt_kappa = []
    gbt_ece = []
    gbt_train_time = []
    gbt_test_time = []
    gbt_probs_labels = []
    storage_dict = {}
    cnt = 1
    for classes in classes_space:
        print(f"{cnt} out of {len(classes_space)}")
        print(classes)
        cnt += 1
        d1 = {}
        # cohen_kappa vs num training samples (gbt)
        for samples in samples_space:
            print(f"Sample size: {samples}")
            l3 = []
            # train data
            # print("init GBT model")
            gbt_model = xgb.XGBClassifier(
                n_estimators=221,
                max_depth=16,
                learning_rate=0.14523012412832884,
                objective='multi:softprob',
                seed=317,
                min_child_weight=2,
                colsample_bytree=0.7571758278257789,
                colsample_bylevel=0.44621372926638425,
                colsample_bynode=0.4129166625194974,
                gamma=0.3279624422858039,
                subsample=0.8281647947326367,
            )
            acc, cohen_kappa, ece, train_time, test_time, test_probs, test_labels, test_preds = run_gbt_image_set(
                gbt_model,
                cifar_train_images,
                cifar_train_labels,
                cifar_test_images,
                cifar_test_labels,
                samples,
                classes,
            )
            print("Accuracy:", acc)
            print("Train time:", train_time)
            print(" ")
            gbt_acc.append(acc)
            gbt_kappa.append(cohen_kappa)
            gbt_ece.append(ece)
            gbt_train_time.append(train_time)
            gbt_test_time.append(test_time)

            # actual_test_labels = []
            # for i in range(len(test_labels)):
            #     actual_test_labels.append(int(classes[test_labels[i]]))

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
    with open(prefix + 'gbt_6hr.pkl', 'wb') as f:
        pickle.dump(gbt_model, f)

    print("gbt finished")
    write_result(prefix + "gbt_acc_6hr.txt", gbt_acc)
    write_result(prefix + "gbt_kappa_6hr.txt", gbt_kappa)
    write_result(prefix + "gbt_ece_6hr.txt", gbt_ece)
    write_result(prefix + "gbt_train_time_6hr.txt", gbt_train_time)
    write_result(prefix + "gbt_test_time_6hr.txt", gbt_test_time)
    write_result(prefix + "gbt_probs_labels_6hr.txt", gbt_probs_labels)
    write_json(prefix + "gbt_acc_6hr.json", gbt_acc)
    write_json(prefix + "gbt_kappa_6hr.json", gbt_kappa)
    write_json(prefix + "gbt_ece_6hr.json", gbt_ece)
    write_json(prefix + "gbt_train_time_6hr.json", gbt_train_time)
    write_json(prefix + "gbt_test_time_6hr.json", gbt_test_time)

def run_naive_rf():
    naive_rf_acc = []
    naive_rf_kappa = []
    naive_rf_ece = []
    naive_rf_train_time = []
    naive_rf_test_time = []
    navie_rf_probs_labels = []
    storage_dict = {}
    cnt = 1
    for classes in classes_space:
        print(f"{cnt} out of {len(classes_space)}")
        print(classes)
        cnt += 1
        d1 = {}
        # cohen_kappa vs num training samples (naive_rf)
        for samples in samples_space:
            print(f"Sample size: {samples}")
            l3 = []
            RF = RandomForestClassifier(
                n_estimators=277, 
                max_depth=14, 
                min_samples_split=5, 
                min_samples_leaf=3, 
                criterion="entropy",
                max_features="log2", 
                max_samples=0.9639616992832429, 
                n_jobs=-1, 
                random_state=317
            )
            acc, cohen_kappa, ece, train_time, test_time, test_probs, test_labels, test_preds = run_rf_image_set(
                RF,
                cifar_train_images,
                cifar_train_labels,
                cifar_test_images,
                cifar_test_labels,
                samples,
                classes,
            )
            print("Accuracy:", acc)
            print("Train time:", train_time)
            print(" ")
            naive_rf_acc.append(acc)
            naive_rf_kappa.append(cohen_kappa)
            naive_rf_ece.append(ece)
            naive_rf_train_time.append(train_time)
            naive_rf_test_time.append(test_time)

            # actual_test_labels = []
            # for i in range(len(test_labels)):
            #     actual_test_labels.append(int(classes[test_labels[i]]))

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
    with open(prefix + 'naive_rf_6hr.pkl', 'wb') as f:
        pickle.dump(RF, f)

    print("naive_rf finished")
    write_result(prefix + "naive_rf_acc_6hr.txt", naive_rf_acc)
    write_result(prefix + "naive_rf_kappa_6hr.txt", naive_rf_kappa)
    write_result(prefix + "naive_rf_ece_6hr.txt", naive_rf_ece)
    write_result(prefix + "naive_rf_train_time_6hr.txt", naive_rf_train_time)
    write_result(prefix + "naive_rf_test_time_6hr.txt", naive_rf_test_time)
    write_result(prefix + "naive_rf_probs_labels_6hr.txt", navie_rf_probs_labels)
    write_json(prefix + "naive_rf_acc_6hr.json", naive_rf_acc)
    write_json(prefix + "naive_rf_kappa_6hr.json", naive_rf_kappa)
    write_json(prefix + "naive_rf_ece_6hr.json", naive_rf_ece)
    write_json(prefix + "naive_rf_train_time_6hr.json", naive_rf_train_time)
    write_json(prefix + "naive_rf_test_time_6hr.json", naive_rf_test_time)
    write_json(prefix + "naive_rf_probs_labels_6hr.json", navie_rf_probs_labels)


def run_cnn32():
    cnn32_acc = []
    cnn32_kappa = []
    cnn32_ece = []
    cnn32_train_time = []
    cnn32_test_time = []
    cnn32_probs_labels = []
    storage_dict = {}
    cnt = 1
    for classes in classes_space:
        print(f"{cnt} out of {len(classes_space)}")
        print(classes)
        cnt += 1
        d1 = {}
        # cohen_kappa vs num training samples (cnn32)
        for samples in samples_space:
            print(f"Sample size: {samples}")
            l3 = []
            # train data
            # train_images, test_valid_images, train_labels, test_valid_labels = split_data(transform=data_transforms)
            # global train_images, test_valid_images, train_labels, test_valid_labels

            cnn32_train_images = trainx.copy()
            cnn32_train_labels = trainy.copy()
            cnn32_test_valid_images = test_validx.copy()
            cnn32_test_valid_labels = test_validy.copy()
            
            cnn32 = SimpleCNN32Filter(len(classes))
            (
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
            ) = prepare_data(
                cnn32_train_images, cnn32_train_labels, cnn32_test_valid_images, cnn32_test_valid_labels, samples, classes
            )

            acc, cohen_kappa, ece, train_time, test_time, test_probs, test_labels, test_preds = run_dn_image_es(
                cnn32,
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
                epochs=30,
                lr=0.1,
                batch=32,
                optimizer_name="adam",
                dampening=0,
                momentum=0,
                weight_decay=0,
            )
            print("Accuracy:", acc)
            print("Train time:", train_time)
            print(" ")
            cnn32_acc.append(acc)
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
    with open(prefix + 'cnn32_6hr.pkl', 'wb') as f:
        pickle.dump(cnn32, f)

    print("cnn32 finished")
    write_result(prefix + "cnn32_acc.txt", cnn32_acc)
    write_result(prefix + "cnn32_kappa.txt", cnn32_kappa)
    write_result(prefix + "cnn32_ece.txt", cnn32_ece)
    write_result(prefix + "cnn32_train_time.txt", cnn32_train_time)
    write_result(prefix + "cnn32_test_time.txt", cnn32_test_time)
    write_result(prefix + "cnn32_probs_labels.txt", cnn32_probs_labels)
    write_json(prefix + "cnn32_acc.json", cnn32_acc)
    write_json(prefix + "cnn32_kappa.json", cnn32_kappa)
    write_json(prefix + "cnn32_ece.json", cnn32_ece)
    write_json(prefix + "cnn32_train_time.json", cnn32_train_time)
    write_json(prefix + "cnn32_test_time.json", cnn32_test_time)
    write_json(prefix + "cnn32_probs_labels.json", cnn32_probs_labels)



def run_cnn32_2l():
    cnn32_2l_acc = []
    cnn32_2l_kappa = []
    cnn32_2l_ece = []
    cnn32_2l_train_time = []
    cnn32_2l_test_time = []
    cnn32_2l_probs_labels = []
    storage_dict = {}
    cnt = 1
    for classes in classes_space:
        print(f"{cnt} out of {len(classes_space)}")
        print(classes)
        cnt += 1
        d1 = {}

        # cohen_kappa vs num training samples (cnn32_2l)
        for samples in samples_space:
            print(f"Sample size: {samples}")
            l3 = []
            # train data
            cnn32_2l_train_images = trainx.copy()
            cnn32_2l_train_labels = trainy.copy()
            cnn32_2l_test_valid_images = test_validx.copy()
            cnn32_2l_test_valid_labels = test_validy.copy()

            cnn32_2l = SimpleCNN32Filter2Layers(len(classes))
            (
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
            ) = prepare_data(
                cnn32_2l_train_images, cnn32_2l_train_labels, cnn32_2l_test_valid_images, cnn32_2l_test_valid_labels, samples, classes
            )
            acc, cohen_kappa, ece, train_time, test_time, test_probs, test_labels, test_preds = run_dn_image_es(
                cnn32_2l,
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
                epochs=176,
                batch=178,
                lr=0.0003003304507514738,
                optimizer_name="sgd",
                # dampening=0,
                # momentum=0,
                weight_decay=0.00035079838014085373,
            )
            print("Accuracy:", acc)
            print("Train time:", train_time)
            print(" ")
            cnn32_2l_acc.append(acc)
            cnn32_2l_kappa.append(cohen_kappa)
            cnn32_2l_ece.append(ece)
            cnn32_2l_train_time.append(train_time)
            cnn32_2l_test_time.append(test_time)

            actual_test_labels = []
            for i in range(len(test_labels)):
                actual_test_labels.append(int(classes[test_labels[i]]))

            sorted_classes = sorted(classes)
            cnn32_2l_probs_labels.append("Classes:" + str(sorted_classes))

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

    with open(prefix +'cnn32_2l_switched_storage_dict.pkl', 'wb') as f:
        pickle.dump(switched_storage_dict, f)

    # save the model
    with open(prefix + 'cnn32_2l_6hr.pkl', 'wb') as f:
        pickle.dump(cnn32_2l, f)

    print("cnn32_2l finished")
    write_result(prefix + "cnn32_2l_acc_6hr.txt", cnn32_2l_acc)
    write_result(prefix + "cnn32_2l_kappa_6hr.txt", cnn32_2l_kappa)
    write_result(prefix + "cnn32_2l_ece_6hr.txt", cnn32_2l_ece)
    write_result(prefix + "cnn32_2l_train_time_6hr.txt", cnn32_2l_train_time)
    write_result(prefix + "cnn32_2l_test_time_6hr.txt", cnn32_2l_test_time)
    write_result(prefix + "cnn32_2l_probs_labels_6hr.txt", cnn32_2l_probs_labels)
    write_json(prefix + "cnn32_2l_acc_6hr.json", cnn32_2l_acc)
    write_json(prefix + "cnn32_2l_kappa_6hr.json", cnn32_2l_kappa)
    write_json(prefix + "cnn32_2l_ece_6hr.json", cnn32_2l_ece)
    write_json(prefix + "cnn32_2l_train_time_6hr.json", cnn32_2l_train_time)
    write_json(prefix + "cnn32_2l_test_time_6hr.json", cnn32_2l_test_time)
    write_json(prefix + "cnn32_2l_probs_labels_6hr.json", cnn32_2l_probs_labels)


def run_cnn32_5l():
    cnn32_5l_acc = []
    cnn32_5l_kappa = []
    cnn32_5l_ece = []
    cnn32_5l_train_time = []
    cnn32_5l_test_time = []
    cnn32_5l_probs_labels = []
    storage_dict = {}

    cnt = 1
    for classes in classes_space:
        print(f"{cnt} out of {len(classes_space)}")
        print(classes)
        cnt += 1
        d1 = {}

        # cohen_kappa vs num training samples (cnn32_5l)
        for samples in samples_space:
            print(f"Sample size: {samples}")
            l3 = []
            # train data
            cifar_trainset = datasets.CIFAR100(
                root="./", train=True, download=True, transform=data_transforms
            )
            cifar_train_labels = np.array(cifar_trainset.targets)

            # test data
            cifar_testset = datasets.CIFAR100(
                root="./", train=False, download=True, transform=data_transforms
            )
            cifar_test_labels = np.array(cifar_testset.targets)

            cnn32_5l = SimpleCNN32Filter5Layers(len(classes))
            train_loader, valid_loader, test_loader = create_loaders_es(
                cifar_train_labels,
                cifar_test_labels,
                classes,
                cifar_trainset,
                cifar_testset,
                samples,
                batch=721,
            )
            acc, cohen_kappa, ece, train_time, test_time, test_probs, test_labels, test_preds = run_dn_image_5l(
                cnn32_5l,
                train_loader,
                valid_loader,
                test_loader,
                epochs=140,
                lr=0.0013471531917165977,
                optimizer_name="adam",
                # dampening=0,
                # momentum=0,
                weight_decay=0.015102902024568963,
            )
            print("Accuracy:", acc)
            print("Train time:", train_time)
            print(" ")
            cnn32_5l_acc.append(acc)
            cnn32_5l_kappa.append(cohen_kappa)
            cnn32_5l_ece.append(ece)
            cnn32_5l_train_time.append(train_time)
            cnn32_5l_test_time.append(test_time)

            actual_test_labels = []
            for i in range(len(test_labels)):
                actual_test_labels.append(int(classes[int(test_labels[i])]))

            sorted_classes = sorted(classes)
            cnn32_5l_probs_labels.append("Classes:" + str(sorted_classes))

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

    with open(prefix +'cnn32_5l_switched_storage_dict.pkl', 'wb') as f:
        pickle.dump(switched_storage_dict, f)

    # save the model
    with open(prefix + 'cnn32_5l_6hr.pkl', 'wb') as f:
        pickle.dump(cnn32_5l, f)

    print("cnn32_5l finished")
    write_result(prefix + "cnn32_5l_acc_6hr2.txt", cnn32_5l_acc)
    write_result(prefix + "cnn32_5l_kappa_6hr2.txt", cnn32_5l_kappa)
    write_result(prefix + "cnn32_5l_ece_6hr2.txt", cnn32_5l_ece)
    write_result(prefix + "cnn32_5l_train_time_6hr2.txt", cnn32_5l_train_time)
    write_result(prefix + "cnn32_5l_test_time_6hr2.txt", cnn32_5l_test_time)
    write_result(prefix + "cnn32_5l_probs_labels_6hr2.txt", cnn32_5l_probs_labels)
    write_json(prefix + "cnn32_5l_acc_6hr2.json", cnn32_5l_acc)
    write_json(prefix + "cnn32_5l_kappa_6hr2.json", cnn32_5l_kappa)
    write_json(prefix + "cnn32_5l_ece_6hr2.json", cnn32_5l_ece)
    write_json(prefix + "cnn32_5l_train_time_6hr2.json", cnn32_5l_train_time)
    write_json(prefix + "cnn32_5l_test_time_6hr2.json", cnn32_5l_test_time)
    write_json(prefix + "cnn32_5l_probs_labels_6hr2.json", cnn32_5l_probs_labels)


def run_resnet18():
    resnet18_acc = []
    resnet18_kappa = []
    resnet18_ece = []
    resnet18_train_time = []
    resnet18_test_time = []
    resnet18_probs_labels = []
    storage_dict = {}
    cnt = 1
    for classes in classes_space:
        print(f"{cnt} out of {len(classes_space)}")
        print(classes)
        cnt += 1
        d1 = {}
        # cohen_kappa vs num training samples (resnet18)
        for samples in samples_space:
            print(f"Sample size: {samples}")
            l3 = []
            # train data
            res_train_images = trainx.copy()
            res_train_labels = trainy.copy()
            res_test_valid_images = testx.copy()
            res_test_valid_labels = testy.copy()


            res = models.resnet18(pretrained=True)
            num_ftrs = res.fc.in_features
            res.fc = nn.Linear(num_ftrs, len(classes))
            (
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
            ) = prepare_data(
                res_train_images, res_train_labels, res_test_valid_images, res_test_valid_labels, samples, classes
            )
            acc, cohen_kappa, ece, train_time, test_time, test_probs, test_labels, test_preds = run_dn_image_es(
                res,
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
                epochs=157,
                batch=376,
                lr=0.0004038452971948572,
                optimizer_name="adam",
                # dampening=0,
                # momentum=0,
                weight_decay=0.0036345023714351886,
            )
            print("Accuracy:", acc)
            print("Train time:", train_time)
            print(" ")
            resnet18_acc.append(acc)
            resnet18_kappa.append(cohen_kappa)
            resnet18_ece.append(ece)
            resnet18_train_time.append(train_time)
            resnet18_test_time.append(test_time)

            actual_test_labels = []
            for i in range(len(test_labels)):
                actual_test_labels.append(int(classes[test_labels[i]]))

            sorted_classes = sorted(classes)
            resnet18_probs_labels.append("Classes:" + str(sorted_classes))

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

    with open(prefix +'resnet18_switched_storage_dict.pkl', 'wb') as f:
        pickle.dump(switched_storage_dict, f)

    # save the model
    with open(prefix + 'resnet18_6hr.pkl', 'wb') as f:
        pickle.dump(res, f)

    print("resnet18 finished")
    write_result(prefix + "resnet18_acc_6hr.txt", resnet18_acc)
    write_result(prefix + "resnet18_kappa_6hr.txt", resnet18_kappa)
    write_result(prefix + "resnet18_ece_6hr.txt", resnet18_ece)
    write_result(prefix + "resnet18_train_time_6hr.txt", resnet18_train_time)
    write_result(prefix + "resnet18_test_time_6hr.txt", resnet18_test_time)
    write_result(prefix + "resnet18_probs_labels_6hr.txt", resnet18_probs_labels)
    write_json(prefix + "resnet18_acc_6hr.json", resnet18_acc)
    write_json(prefix + "resnet18_kappa_6hr.json", resnet18_kappa)
    write_json(prefix + "resnet18_ece_6hr.json", resnet18_ece)
    write_json(prefix + "resnet18_train_time_6hr.json", resnet18_train_time)
    write_json(prefix + "resnet18_test_time_6hr.json", resnet18_test_time)
    write_json(prefix + "resnet18_probs_labels_6hr.json", resnet18_probs_labels)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    # Example usage: python cifar_100.py -m 90
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", help="class number")
    args = parser.parse_args()
    n_classes = int(args.m)
    prefix = args.m + "_class/"
    samples_space = np.geomspace(100, 9000, num=8, dtype=int)

    nums = list(range(100))
    random.shuffle(nums)
    classes_space = list(combinations_45(nums, n_classes))

    # normalize
    # scale = np.mean(np.arange(0, 256))
    # normalize = lambda x: (x - scale) / scale
    # New normalized method by Ziyan Li 
    normalize = lambda x: x / 255.0

    # for CNNs
    # data_transforms = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # )

    # for resnet
    data_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


    # train data
    cifar_trainset = datasets.CIFAR100(
        root="./", train=True, download=True, transform=None
    )
    cifar_train_images = normalize(cifar_trainset.data)
    cifar_train_labels = np.array(cifar_trainset.targets)

    # test data
    cifar_testset = datasets.CIFAR100(
        root="./", train=False, download=True, transform=None
    )
    cifar_test_images = normalize(cifar_testset.data)
    cifar_test_labels = np.array(cifar_testset.targets)

    images = np.concatenate((cifar_train_images, cifar_test_images))
    labels = np.concatenate((cifar_train_labels, cifar_test_labels))

    # shuffle data
    indices = np.arange(images.shape[0])
    np.random.shuffle(indices)
    images = images[indices]
    labels = labels[indices]

    # split data
    train_images, test_valid_images, train_labels, test_valid_labels = train_test_split(
        images, labels, test_size=0.5, random_state=317
    )

    test_images, valid_images, test_labels, valid_labels = train_test_split(
        test_valid_images, test_valid_labels, test_size=0.5, random_state=317
    )

    cifar_train_images = train_images.copy().reshape(-1, 32 * 32 * 3)
    cifar_train_labels = train_labels.copy()
    cifar_test_images = test_images.copy().reshape(-1, 32 * 32 * 3)
    cifar_test_labels = test_labels.copy()
    cifar_valid_images = valid_images.copy().reshape(-1, 32 * 32 * 3)
    cifar_valid_labels = valid_labels.copy()

    trainx = train_images.copy()
    trainy = train_labels.copy()
    testx = test_images.copy()
    testy = test_labels.copy()
    validx = valid_images.copy()
    validy = valid_labels.copy()
    test_validx = test_valid_images.copy()
    test_validy = test_valid_labels.copy()

    # print("Running GBT tuning \n")
    # run_GBT()

    # print("Running RF tuning \n")
    # run_naive_rf()

    # print("Running CNN32 tuning \n")
    # run_cnn32()

    # print("Running CNN32_2l tuning \n")
    # run_cnn32_2l()

    print("Running CNN32_5l tuning \n")
    run_cnn32_5l()

    # print("Running Resnet18 tuning \n")
    # run_resnet18()
