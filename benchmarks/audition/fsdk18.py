"""
Coauthors: Haoyin Xu
           Yu-Chung Peng
           Madi Kusmanov
           Adway Kanhere
"""
from toolbox import *
import argparse
import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import warnings


logger = logging.getLogger(tune.__name__)
logger.setLevel(
    level=logging.CRITICAL
)  # Reduce the number of Ray warnings that are not relevant here.

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_naive_rf():
    naive_rf_kappa = []
    naive_rf_ece = []
    naive_rf_train_time = []
    naive_rf_test_time = []
    for classes in classes_space:

        # cohen_kappa vs num training samples (naive_rf)
        for samples in samples_space:
            # train data
            RF = RandomForestClassifier(n_estimators=100, n_jobs=-1)
            cohen_kappa, ece, train_time, test_time = run_rf_image_set(
                RF,
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

    print("naive_rf finished")
    write_result(prefix + "naive_rf_kappa.txt", naive_rf_kappa)
    write_result(prefix + "naive_rf_ece.txt", naive_rf_ece)
    write_result(prefix + "naive_rf_train_time.txt", naive_rf_train_time)
    write_result(prefix + "naive_rf_test_time.txt", naive_rf_test_time)


def run_cnn32():
    cnn32_kappa = []
    cnn32_ece = []
    cnn32_train_time = []
    cnn32_test_time = []
    cnn32_tune_time = []
    best_objs = []
    best_params = []
    for samples in samples_space:

        # cohen_kappa vs num training samples (cnn32)
        for classes in classes_space:
            # train data
            train_images = trainx.copy()
            train_labels = trainy.copy()
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

            start_time = time.perf_counter()
            arm, best_obj = run_dn_image_es(
                "cnn32",
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
                classes,
            )
            end_time = time.perf_counter()
            tune_time = end_time - start_time

            best_objs.append(best_obj)
            best_params.append(arm)

            combined_train_valid_data = torch.cat((train_images, valid_images), dim=0)
            combined_train_valid_labels = torch.cat((train_labels, valid_labels), dim=0)

            cnn32_final = SimpleCNN32Filter(len(classes))

            start_time = time.perf_counter()
            model_retrain_aftertune = training_net(
                cnn32_final,
                arm.parameters,
                combined_train_valid_data,
                combined_train_valid_labels,
            )
            end_time = time.perf_counter()
            train_time = end_time - start_time

            final_ck, final_ece, test_time = evaluate_net_final(
                model_retrain_aftertune, test_images, test_labels, 60, dev=device
            )

            cnn32_kappa.append(final_ck)
            cnn32_ece.append(final_ece)
            cnn32_tune_time.append(tune_time)
            cnn32_train_time.append(train_time)
            cnn32_test_time.append(test_time)

    print("cnn32 finished")
    write_result(prefix + "cnn32_bestparams.txt", best_params)
    write_result(prefix + "cnn32_kappa.txt", cnn32_kappa)
    write_result(prefix + "cnn32_ece.txt", cnn32_ece)
    write_result(prefix + "cnn32_tune_time.txt", cnn32_tune_time)
    write_result(prefix + "cnn32_train_time.txt", cnn32_train_time)
    write_result(prefix + "cnn32_test_time.txt", cnn32_test_time)


def run_cnn32_2l():
    cnn32_2l_kappa = []
    cnn32_2l_ece = []
    cnn32_2l_train_time = []
    cnn32_2l_tune_time = []
    cnn32_2l_test_time = []
    best_objs = []
    best_params = []
    for samples in samples_space:

        # cohen_kappa vs num training samples (cnn32_2l)
        for classes in classes_space:
            # train data
            train_images = trainx.copy()
            train_labels = trainy.copy()
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

            start_time = time.perf_counter()
            arm, best_obj = run_dn_image_es(
                "cnn32_2l",
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
                classes,
            )
            end_time = time.perf_counter()
            tune_time = end_time - start_time

            best_objs.append(best_obj)
            best_params.append(arm)

            combined_train_valid_data = torch.cat((train_images, valid_images), dim=0)
            combined_train_valid_labels = torch.cat((train_labels, valid_labels), dim=0)

            cnn32_2l_final = SimpleCNN32Filter2Layers(len(classes))

            start_time = time.perf_counter()
            model_retrain_aftertune = training_net(
                cnn32_2l_final,
                arm.parameters,
                combined_train_valid_data,
                combined_train_valid_labels,
            )
            end_time = time.perf_counter()
            train_time = end_time - start_time

            final_ck, final_ece, test_time = evaluate_net_final(
                model_retrain_aftertune, test_images, test_labels, 60, dev=device
            )

            cnn32_2l_kappa.append(final_ck)
            cnn32_2l_ece.append(final_ece)
            cnn32_2l_tune_time.append(tune_time)
            cnn32_2l_train_time.append(train_time)
            cnn32_2l_test_time.append(test_time)

    print("cnn32_2l finished")
    write_result(prefix + "cnn32_2l_bestparams.txt", best_params)
    write_result(prefix + "cnn32_2l_kappa.txt", cnn32_2l_kappa)
    write_result(prefix + "cnn32_2l_ece.txt", cnn32_2l_ece)
    write_result(prefix + "cnn32_2l_tune_time.txt", cnn32_2l_tune_time)
    write_result(prefix + "cnn32_2l_train_time.txt", cnn32_2l_train_time)
    write_result(prefix + "cnn32_2l_test_time.txt", cnn32_2l_test_time)


def run_cnn32_5l():
    cnn32_5l_kappa = []
    cnn32_5l_ece = []
    cnn32_5l_tune_time = []
    cnn32_5l_train_time = []
    cnn32_5l_test_time = []
    best_objs = []
    best_params = []
    for samples in samples_space:

        # cohen_kappa vs num training samples (cnn32_5l)
        for classes in classes_space:
            # train data
            train_images = trainx.copy()
            train_labels = trainy.copy()
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

            start_time = time.perf_counter()
            arm, best_obj = run_dn_image_es(
                "cnn32_5l",
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
                classes,
            )
            end_time = time.perf_counter()
            tune_time = end_time - start_time

            best_objs.append(best_obj)
            best_params.append(arm)

            combined_train_valid_data = torch.cat((train_images, valid_images), dim=0)
            combined_train_valid_labels = torch.cat((train_labels, valid_labels), dim=0)

            cnn32_5l_final = SimpleCNN32Filter5Layers(len(classes))

            start_time = time.perf_counter()
            model_retrain_aftertune = training_net(
                cnn32_5l_final,
                arm.parameters,
                combined_train_valid_data,
                combined_train_valid_labels,
            )
            end_time = time.perf_counter()
            train_time = end_time - start_time

            final_ck, final_ece, test_time = evaluate_net_final(
                model_retrain_aftertune, test_images, test_labels, 60, dev=device
            )

            cnn32_5l_kappa.append(final_ck)
            cnn32_5l_ece.append(final_ece)
            cnn32_5l_tune_time.append(tune_time)
            cnn32_5l_train_time.append(train_time)
            cnn32_5l_test_time.append(test_time)

    print("cnn32_5l finished")
    write_result(prefix + "cnn32_5l_bestparams.txt", best_params)
    write_result(prefix + "cnn32_5l_kappa.txt", cnn32_5l_kappa)
    write_result(prefix + "cnn32_5l_ece.txt", cnn32_5l_ece)
    write_result(prefix + "cnn32_5l_tune_time.txt", cnn32_5l_tune_time)
    write_result(prefix + "cnn32_5l_train_time.txt", cnn32_5l_train_time)
    write_result(prefix + "cnn32_5l_test_time.txt", cnn32_5l_test_time)


def run_resnet18():
    resnet18_kappa = []
    resnet18_ece = []
    resnet18_tune_time = []
    resnet18_train_time = []
    resnet18_test_time = []
    best_objs = []
    best_params = []

    for samples in samples_space:
        # cohen_kappa vs num training samples (resnet18)
        for classes in classes_space:
            # train data
            train_images = trainx.copy()
            train_labels = trainy.copy()
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

            start_time = time.perf_counter()
            arm, best_obj = run_dn_image_es(
                "resnet",
                train_images,
                train_labels,
                valid_images,
                valid_labels,
                test_images,
                test_labels,
                classes,
            )
            end_time = time.perf_counter()
            tune_time = end_time - start_time

            best_objs.append(best_obj)
            best_params.append(arm)

            combined_train_valid_data = torch.cat((train_images, valid_images), dim=0)
            combined_train_valid_labels = torch.cat((train_labels, valid_labels), dim=0)

            resnet_final = models.resnet18(pretrained=True)
            num_ftrs = resnet_final.fc.in_features
            resnet_final.fc = nn.Linear(num_ftrs, len(classes))

            start_time = time.perf_counter()
            model_retrain_aftertune = training_net(
                resnet_final,
                arm.parameters,
                combined_train_valid_data,
                combined_train_valid_labels,
            )
            end_time = time.perf_counter()
            train_time = end_time - start_time

            final_ck, final_ece, test_time = evaluate_net_final(
                model_retrain_aftertune, test_images, test_labels, 60, dev=device
            )

            resnet18_kappa.append(final_ck)
            resnet18_ece.append(final_ece)
            resnet18_tune_time.append(tune_time)
            resnet18_train_time.append(train_time)
            resnet18_test_time.append(test_time)

    print("resnet18 finished")
    write_result(prefix + "resnet18_bestparams.txt", best_params)
    write_result(prefix + "resnet18_kappa.txt", resnet18_kappa)
    write_result(prefix + "resnet18_ece.txt", resnet18_ece)
    write_result(prefix + "resnet18_tune_time.txt", resnet18_tune_time)
    write_result(prefix + "resnet18_train_time.txt", resnet18_train_time)
    write_result(prefix + "resnet18_test_time.txt", resnet18_test_time)


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

    # Preprocess and subset the data
    path_recordings, labels_chosen, get_labels = preprocessdataset(
        str(args.data), str(args.labels)
    )

    # data is normalized upon loading
    # load dataset
    x_spec, y_number = load_fsdk18(
        path_recordings, labels_chosen, get_labels, feature_type
    )
    nums = list(range(18))
    if n_classes == 3:
        samples_space = np.geomspace(10, 450, num=6, dtype=int)
    else:
        samples_space = np.geomspace(10, 1200, num=6, dtype=int)
    # define path, samples space and number of class combinations
    if feature_type == "melspectrogram":
        prefix = args.m + "_class_mel/"
    elif feature_type == "spectrogram":
        prefix = args.m + "_class/"
    elif feature_type == "mfcc":
        prefix = args.m + "_class_mfcc/"

    # create list of classes with const random seed
    np.random.shuffle(nums)
    classes_space = list(combinations_45(nums, n_classes))

    # scale the data
    # x_spec = x_spec[:5400] #reshape x_spec by Ziyan for testing, orginial shape was (11073, 32, 32)
    print(x_spec.shape)
    x_spec = scale(x_spec.reshape(5400, -1), axis=1).reshape(5400, 32, 32)
    y_number = np.array(y_number)
    # y_number = y_number[:5400] #reshape x_spec by Ziyan for testing, orginial shape was (11073, 32, 32)

    # need to take train/valid/test equally from each class
    trainx, testx, trainy, testy = train_test_split(
        x_spec,
        y_number,
        shuffle=True,
        test_size=0.50,
        train_size=0.50,
        stratify=y_number,
    )

    # 3000 samples, 80% train is 2400 samples, 20% test
    fsdk18_train_images = trainx.reshape(-1, 32 * 32)
    fsdk18_train_labels = trainy.copy()
    # reshape in 2d array
    fsdk18_test_images = testx.reshape(-1, 32 * 32)
    fsdk18_test_labels = testy.copy()

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
