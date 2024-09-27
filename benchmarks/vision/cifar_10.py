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

    for classes in classes_space:
        print("classes: ", classes)
        # print(classes)
        d1 = {}
        # cohen_kappa vs num training samples (gbt)
        for samples in samples_space:
            print("samples: ", samples)
            l3 = []
            # train data
            # print("init GBT model")
            gbt_model = xgb.XGBClassifier(
                n_estimators=1045,
                max_depth=14,
                min_child_weight=6,
                learning_rate=0.015531869905407784,
                objective='multi:softprob',
                seed=317,
                colsample_bytree=0.8658137268853405,
                colsample_bylevel=0.5062758127741084,
                colsample_bynode=0.4623908082630023,
                gamma=0.5544812184195786,
                subsample=0.765961974698761,


                # colsample_bytree=0.6702264438270331,
                # colsample_bylevel=0.8006325564606935,
                # colsample_bynode=0.6063110478838847,
                # gamma=0.11819655769629316,
                # subsample=0.7185159768996707,
                # n_estimators=773,
                # max_depth=4,
                # min_child_weight=8,
                # learning_rate=0.24175598362284983,

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
            print("Accuracy: ", acc)
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
    with open(prefix + 'gbt_org.pkl', 'wb') as f:
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

    for classes in classes_space:
        print("classes: ", classes)
        d1 = {}
        # cohen_kappa vs num training samples (naive_rf)
        for samples in samples_space:
            print("samples: ", samples)
            l3 = []
            RF = RandomForestClassifier(n_estimators=1038, min_samples_leaf=1, min_samples_split=3,max_depth=18, criterion='gini', max_features=None, max_samples=0.8070152333771876, n_jobs=-1, random_state=317)
            acc, cohen_kappa, ece, train_time, test_time, test_probs, test_labels, test_preds = run_rf_image_set(
                RF,
                cifar_train_images,
                cifar_train_labels,
                cifar_test_images,
                cifar_test_labels,
                samples,
                classes,
            )
            print("Accuracy: ", acc)
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
    with open(prefix + 'naive_rf_org.pkl', 'wb') as f:
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
    for classes in classes_space:
        d1 = {}
        print("classes:", classes)
        # cohen_kappa vs num training samples (cnn32)
        for samples in samples_space:
            print("samples:", samples)
            l3 = []
            # train data
            # train_images, test_valid_images, train_labels, test_valid_labels = split_data(transform=data_transforms)
            # global train_images, test_valid_images, train_labels, test_valid_labels

            cnn32_train_images = cifar_train_img.copy()
            cnn32_train_labels = cifar_train_lab.copy()
            cnn32_test_valid_images = cifar_test_img.copy()
            cnn32_test_valid_labels = cifar_test_lab.copy()

            
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
                epochs=160,
                batch=1013,
                lr=0.00012048935640559353,
                optimizer_name="adam",
                # dampening=0.6591321889550715,
                # momentum=0.9027870841688489,
                weight_decay=0.009961335002167393,
            )
            print("accuracy:", acc)
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
    with open(prefix + 'cnn32.pkl', 'wb') as f:
        pickle.dump(cnn32, f)

    print("cnn32 finished")
    write_result(prefix + "cnn32_acc_2hr.txt", cnn32_acc)
    write_result(prefix + "cnn32_kappa_2hr.txt", cnn32_kappa)
    write_result(prefix + "cnn32_ece_2hr.txt", cnn32_ece)
    write_result(prefix + "cnn32_train_time_2hr.txt", cnn32_train_time)
    write_result(prefix + "cnn32_test_time_2hr.txt", cnn32_test_time)
    write_result(prefix + "cnn32_probs_labels_2hr.txt", cnn32_probs_labels)
    write_json(prefix + "cnn32_acc_2hr.json", cnn32_acc)
    write_json(prefix + "cnn32_kappa_2hr.json", cnn32_kappa)
    write_json(prefix + "cnn32_ece_2hr.json", cnn32_ece)
    write_json(prefix + "cnn32_train_time_2hr.json", cnn32_train_time)
    write_json(prefix + "cnn32_test_time_2hr.json", cnn32_test_time)
    write_json(prefix + "cnn32_probs_labels_2hr.json", cnn32_probs_labels)


def run_cnn32_2l():
    cnn32_2l_acc = []
    cnn32_2l_kappa = []
    cnn32_2l_ece = []
    cnn32_2l_train_time = []
    cnn32_2l_test_time = []
    cnn32_2l_probs_labels = []
    storage_dict = {}
    for classes in classes_space:
        print("classes:", classes)
        d1 = {}

        # cohen_kappa vs num training samples (cnn32_2l)
        for samples in samples_space:
            print("samples:", samples)
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
                epochs=162,
                batch=16,
                lr=0.00017042781035706246,
                optimizer_name="adam",
                # dampening=0.15737988780906453,
                # momentum=0.7902936875443869,
                # weight_decay=0.0016764535333438713,
            )
            print("accuracy:", acc)
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
    with open(prefix + 'cnn32_2l.pkl', 'wb') as f:
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

    for classes in classes_space:
        d1 = {}
        print("classes:", classes)
        # cohen_kappa vs num training samples (cnn32_5l)
        for samples in samples_space:
            print("samples:", samples)
            l3 = []
            # train data
            cifar_trainset = datasets.CIFAR10(
                root="./", train=True, download=True, transform=data_transforms
            )
            cifar_train_labels = np.array(cifar_trainset.targets)

            # test data
            cifar_testset = datasets.CIFAR10(
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
                batch=17
            )
            acc, cohen_kappa, ece, train_time, test_time, test_probs, test_labels, test_preds = run_dn_image_5l(
                cnn32_5l,
                train_loader,
                valid_loader,
                test_loader,
                epochs=100,
                lr=0.00013334064818344654,
                optimizer_name="adam",
                # momentum=0.9,
                weight_decay=0.07959524081808507,
                # dampening=0,
            )
            print("accuarcy:", acc)
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
    with open(prefix + 'cnn32_5l.pkl', 'wb') as f:
        pickle.dump(cnn32_5l, f)

    print("cnn32_5l finished")
    write_result(prefix + "cnn32_5l_acc_6hr.txt", cnn32_5l_acc)
    write_result(prefix + "cnn32_5l_kappa_6hr.txt", cnn32_5l_kappa)
    write_result(prefix + "cnn32_5l_ece_6hr.txt", cnn32_5l_ece)
    write_result(prefix + "cnn32_5l_train_time_6hr.txt", cnn32_5l_train_time)
    write_result(prefix + "cnn32_5l_test_time_6hr.txt", cnn32_5l_test_time)
    write_result(prefix + "cnn32_5l_probs_labels_6hr.txt", cnn32_5l_probs_labels)
    write_json(prefix + "cnn32_5l_acc_6hr.json", cnn32_5l_acc)
    write_json(prefix + "cnn32_5l_kappa_6hr.json", cnn32_5l_kappa)
    write_json(prefix + "cnn32_5l_ece_6hr.json", cnn32_5l_ece)
    write_json(prefix + "cnn32_5l_train_time_6hr.json", cnn32_5l_train_time)
    write_json(prefix + "cnn32_5l_test_time_6hr.json", cnn32_5l_test_time)
    write_json(prefix + "cnn32_5l_probs_labels_6hr.json", cnn32_5l_probs_labels)


def run_resnet18():
    resnet18_acc = []
    resnet18_kappa = []
    resnet18_ece = []
    resnet18_train_time = []
    resnet18_test_time = []
    resnet18_probs_labels = []
    storage_dict = {}
    for classes in classes_space:
        print("classes:", classes)
        d1 = {}
        # cohen_kappa vs num training samples (resnet18)
        for samples in samples_space:
            print("samples:", samples)
            l3 = []
            # train data
            res_train_images = trainx.copy()
            res_train_labels = trainy.copy()
            res_test_valid_images = test_validx.copy()
            res_test_valid_labels = test_validy.copy()

            total_images = images.copy()
            total_labels = labels.copy()


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
                epochs=160,
                batch=16,
                lr=0.0008865001733821951,
                optimizer_name="adam",
                # dampening=0.8295654584238159,
                # momentum=0.9666948703632617,
                weight_decay=0.00024133293809938924,
            )
            print("Accuracy:", acc)
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
    with open(prefix + 'resnet18.pkl', 'wb') as f:
        pickle.dump(res, f)

    print("resnet18 finished")
    write_result(prefix + "resnet18_acc_6hr2.txt", resnet18_acc)
    write_result(prefix + "resnet18_kappa_6hr2.txt", resnet18_kappa)
    write_result(prefix + "resnet18_ece_6hr2.txt", resnet18_ece)
    write_result(prefix + "resnet18_train_time_6hr2.txt", resnet18_train_time)
    write_result(prefix + "resnet18_test_time_6hr2.txt", resnet18_test_time)
    write_result(prefix + "resnet18_probs_labels_6hr2.txt", resnet18_probs_labels)
    write_json(prefix + "resnet18_acc_6hr2.json", resnet18_acc)
    write_json(prefix + "resnet18_kappa_6hr2.json", resnet18_kappa)
    write_json(prefix + "resnet18_ece_6hr2.json", resnet18_ece)
    write_json(prefix + "resnet18_train_time_6hr2.json", resnet18_train_time)
    write_json(prefix + "resnet18_test_time_6hr2.json", resnet18_test_time)
    write_json(prefix + "resnet18_probs_labels_6hr2.json", resnet18_probs_labels)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    # Example usage: python cifar_10.py -m 3
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", help="class number")
    args = parser.parse_args()
    n_classes = int(args.m)
    prefix = args.m + "_class/"
    samples_space = np.geomspace(10, 9000, num=8, dtype=int)

    nums = list(range(10))
    random.shuffle(nums)
    classes_space = list(combinations_45(nums, n_classes))
    # print()

    # normalize
    # scale = np.mean(np.arange(0, 256))
    # normalize = lambda x: (x - scale) / scale
    # New normalized method by Ziyan Li 
    normalize = lambda x: x / 255.0

    # For CNN32
    # data_transforms = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # )

    # For ResNet
    data_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    # data_transforms = transforms.Compose(
    #     [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    # )

    # train data
    cifar_trainset = datasets.CIFAR10(
        root="./", train=True, download=True, transform=None
    )
    cifar_train_images = normalize(cifar_trainset.data)
    cifar_train_labels = np.array(cifar_trainset.targets)

    # test data
    cifar_testset = datasets.CIFAR10(
        root="./", train=False, download=True, transform=None
    )
    cifar_test_images = normalize(cifar_testset.data)
    cifar_test_labels = np.array(cifar_testset.targets)

    print("train/test ratio:", len(cifar_train_labels) / len(cifar_test_labels))

    # cifar_train_images = cifar_train_images.reshape(-1, 32 * 32 * 3)
    # cifar_test_images = cifar_test_images.reshape(-1, 32 * 32 * 3)

    cifar_train_img = cifar_train_images.copy()
    cifar_train_lab = cifar_train_labels.copy()
    cifar_test_img = cifar_test_images.copy()
    cifar_test_lab = cifar_test_labels.copy()

    # concatenate data
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

    print("train/test ratio after split:", len(trainx) / len(testx))




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
