"""
Coauthors: Haoyin Xu
           Yu-Chung Peng
           Madi Kusmanov
           Jayanta Dey
           Adway Kanhere
"""
import time
import os
import cv2
import librosa
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import torchaudio.transforms as trans
import torchvision.models as models


# import ax
# from ax.plot.trace import optimization_trace_single_method
# from ax.service.managed_loop import optimize
# from ax.service.ax_client import AxClient

from ray import tune
from ray.tune import report
from ray.tune.search.ax import AxSearch
from ray.tune.schedulers import ASHAScheduler


def preprocessdataset(data_folder, labels_file):
    """
    This function performs pre-preocessing of the FSDKaggle18 dataset such that:
    - Of all the labels, select only those which contain 300 samples per label
    - Extract the corresponding audio files for the selected labels
    --------------------------------------------
    Input parameters:
        data_folder: str
          Path to the folder containing the raw audio files
        labels_file: str
          Path to the csv file containing the labels
    Returns:
        path_recordings: list
          list containing the full path for each selected raw audio file
        labels_chosen: pandas DataFrame
          DataFrame of ground truth labels and metadata of all the selected raw audio files
        get_labels: array
          array of encoded unique label numbers
    """

    train_folder = data_folder
    train_label = pd.read_csv(labels_file)

    # select subset of data that only contains 300 samples per class
    labels_chosen = train_label[
        train_label["label"].map(train_label["label"].value_counts() >= 300)
    ]

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
            "Bark": 2,
            "Bass_drum": 3,
            "Burping_or_eructation": 4,
            "Bus": 5,
            "Cello": 6,
            "Chime": 7,
            "Clarinet": 8,
            "Computer_keyboard": 9,
            "Cough": 10,
            "Cowbell": 11,
            "Double_bass": 12,
            "Drawer_open_or_close": 13,
            "Electric_piano": 14,
            "Fart": 15,
            "Finger_snapping": 16,
            "Fireworks": 17,
            "Flute": 18,
            "Glockenspiel": 19,
            "Gong": 20,
            "Gunshot_or_gunfire": 21,
            "Harmonica": 22,
            "Hi-hat": 23,
            "Keys_jangling": 24,
            "Knock": 25,
            "Laughter": 26,
            "Meow": 27,
            "Microwave_oven": 28,
            "Oboe": 29,
            "Saxophone": 30,
            "Scissors": 31,
            "Shatter": 32,
            "Snare_drum": 33,
            "Squeak": 34,
            "Tambourine": 35,
            "Tearing": 36,
            "Telephone": 37,
            "Trumpet": 38,
            "Violin_or_fiddle": 39,
            "Writing": 40,
        }

    # encode labels to integers
    get_labels = labels_chosen["label"].replace(labels_to_index).to_list()
    labels_chosen = labels_chosen.reset_index()

    return path_recordings, labels_chosen, get_labels


def load_fsdk18(path_recordings, labels_file, label_arr, feature_type="spectrogram"):

    audio_data = []  # audio data
    x_audio = []  # STFT spectrogram
    x_audio_mini = []  # resized image, 32*32
    y_number = []  # label of number  # label of speaker
    if feature_type == "spectrogram":
        a = trans.Spectrogram(n_fft=128, normalized=True)
    elif feature_type == "melspectrogram":
        a = trans.MelSpectrogram(n_fft=128, normalized=True)
    elif feature_type == "mfcc":
        a = trans.MFCC(n_mfcc=128)
    for i in path_recordings:
        x, sr = librosa.load(i, sr=44100)
        i = i[-12:]
        x_stft_db = a(torch.tensor(x)).numpy()
        # Convert an amplitude spectrogram to dB-scaled spectrogram
        x_stft_db_mini = cv2.resize(x_stft_db, (32, 32))  # Resize into 32 by 32
        get_label_location = int(
            labels_file.fname.index[labels_file["fname"] == i].to_numpy()
        )

        y_n = label_arr[get_label_location]  # label number
        audio_data.append(x)
        x_audio.append(x_stft_db)
        x_audio_mini.append(x_stft_db_mini)
        y_number.append(y_n)

    x_audio_mini = np.array(x_audio_mini)
    y_number = np.array(y_number).astype(int)

    return x_audio_mini, y_number


def load_spoken_digit(path_recordings, feature_type="spectrogram"):
    file = os.listdir(path_recordings)

    audio_data = []  # audio data
    x_spec = []  # STFT spectrogram
    x_spec_mini = []  # resized image, 32*32
    y_number = []  # label of number
    y_speaker = []  # label of speaker
    if feature_type == "spectrogram":
        a = trans.Spectrogram(n_fft=128, normalized=True)
    elif feature_type == "melspectrogram":
        a = trans.MelSpectrogram(n_fft=128, normalized=True)
    elif feature_type == "mfcc":
        a = trans.MFCC(n_mfcc=128)
    for i in file:
        x, sr = librosa.load(path_recordings + i, sr=8000)
        x_stft_db = a(torch.tensor(x)).numpy()
        # Convert an amplitude spectrogram to dB-scaled spectrogram
        x_stft_db_mini = cv2.resize(x_stft_db, (32, 32))  # Resize into 32 by 32
        y_n = i[0]  # number
        y_s = i[2]  # first letter of speaker's name

        audio_data.append(x)
        x_spec.append(x_stft_db)
        x_spec_mini.append(x_stft_db_mini)
        y_number.append(y_n)
        y_speaker.append(y_s)

    x_spec_mini = np.array(x_spec_mini)
    y_number = np.array(y_number).astype(int)
    y_speaker = np.array(y_speaker)

    return x_spec_mini, y_number


class SimpleCNN32Filter(nn.Module):
    """
    Defines a simple CNN arhcitecture
    """

    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=10, stride=2)
        self.fc1 = nn.Linear(144 * 32, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 144 * 32)
        x = self.fc1(x)
        return x


class SimpleCNN32Filter2Layers(nn.Module):
    """
    Define a simple CNN arhcitecture with 2 layers
    """

    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(12 * 12 * 32, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        b = x.shape[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(b, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleCNN32Filter5Layers(nn.Module):
    """
    Define a simple CNN arhcitecture with 5 layers
    """

    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(8192, 200)
        self.fc2 = nn.Linear(200, num_classes)
        self.maxpool = nn.MaxPool2d((2, 2))
        self.bn = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x):
        b = x.shape[0]
        x = F.relu(self.bn(self.conv1(x)))
        x = F.relu(self.bn(self.conv2(x)))
        x = self.maxpool(x)
        x = F.relu(self.bn2(self.conv3(x)))
        x = F.relu(self.bn2(self.conv4(x)))
        x = F.relu(self.bn3(self.conv5(x)))
        x = self.maxpool(x)
        x = x.view(b, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def write_result(filename, acc_ls):
    """Writes results to specified text file"""
    output = open(filename, "w")
    for acc in acc_ls:
        output.write(str(acc) + "\n")


def combinations_45(iterable, r):
    """Extracts 45 combinations from given list"""
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    count = 0
    while count < 44:
        count += 1
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        yield tuple(pool[i] for i in indices)


def get_ece(predicted_posterior, predicted_label, true_label, num_bins=40):
    """
    Return expected calibration error (ECE)
    """
    poba_hist = []
    accuracy_hist = []
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


def run_rf_image_set(
    model,
    train_images,
    train_labels,
    test_images,
    test_labels,
    samples,
    classes,
):
    """
    Peforms multiclass predictions for a random forest classifier
    with fixed total samples
    """
    num_classes = len(classes)
    partitions = np.array_split(np.array(range(samples)), num_classes)

    # Obtain only train images and labels for selected classes
    image_ls = []
    label_ls = []
    i = 0
    for cls in classes:
        class_idx = np.argwhere(train_labels == cls).flatten()
        np.random.shuffle(class_idx)
        class_img = train_images[class_idx[: len(partitions[i])]]
        image_ls.append(class_img)
        label_ls.append(np.repeat(cls, len(partitions[i])))
        i += 1

    print(len(image_ls))
    train_images = np.concatenate(image_ls)
    train_labels = np.concatenate(label_ls)
    train_labels = train_labels[:len(train_images)] 
    # Sometimes the shape of the train_labels would be differ from the train_images, so I just changed the shape of train_labels Ziyan

    # Obtain only test images and labels for selected classes
    image_ls = []
    label_ls = []
    for cls in classes:
        image_ls.append(test_images[test_labels == cls])
        label_ls.append(np.repeat(cls, np.sum(test_labels == cls)))

    test_images = np.concatenate(image_ls)
    test_labels = np.concatenate(label_ls)

    # Train the model
    start_time = time.perf_counter()
    model.fit(train_images, train_labels)
    end_time = time.perf_counter()
    train_time = end_time - start_time

    # Test the model
    start_time = time.perf_counter()
    test_preds = model.predict(test_images)
    end_time = time.perf_counter()
    test_time = end_time - start_time

    test_probs = model.predict_proba(test_images)

    return (
        cohen_kappa_score(test_labels, test_preds),
        get_ece(test_probs, test_preds, test_labels),
        train_time,
        test_time,
    )


# def run_dn_image_es(
#     model,
#     train_data,
#     train_labels,
#     valid_data,
#     valid_labels,
#     test_data,
#     test_labels,
#     epochs=30,
#     lr=0.001,
#     batch=60,
# ):
#     """
#     Peforms multiclass predictions for a deep network classifier with set number
#     of samples and early stopping
#     """
#     # define model
#     dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.to(dev)
#     # loss and optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
#     # early stopping setup
#     prev_loss = float("inf")
#     flag = 0
#     start_time = time.perf_counter()
#     for epoch in range(epochs):  # loop over the dataset multiple times
#         model.train()
#         for i in range(0, len(train_data), batch):
#             # get the inputs
#             inputs = train_data[i : i + batch].to(dev)
#             labels = train_labels[i : i + batch].to(dev)
#             # zero the parameter gradients
#             optimizer.zero_grad()
#
#             # forward + backward + optimize
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#         # test generalization error for early stopping
#         model.eval()
#         cur_loss = 0
#         with torch.no_grad():
#             for i in range(0, len(valid_data), batch):
#                 # get the inputs
#                 inputs = valid_data[i : i + batch].to(dev)
#                 labels = valid_labels[i : i + batch].to(dev)
#
#                 # forward
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 cur_loss += loss
#         # early stop if 3 epochs in a row no loss decrease
#         if cur_loss < prev_loss:
#             prev_loss = cur_loss
#             flag = 0
#         else:
#             flag += 1
#             if flag >= 3:
#                 print("early stopped at epoch: ", epoch)
#                 break
#     end_time = time.perf_counter()
#     train_time = end_time - start_time
#
#     # test the model
#     model.eval()
#     prob_cal = nn.Softmax(dim=1)
#     start_time = time.perf_counter()
#     test_preds = []
#     with torch.no_grad():
#         for i in range(0, len(test_data), batch):
#             inputs = test_data[i : i + batch].to(dev)
#             labels = test_labels[i : i + batch].to(dev)
#
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs.data, 1)
#             test_preds = np.concatenate((test_preds, predicted.tolist()))
#
#             test_prob = prob_cal(outputs)
#             if i == 0:
#                 test_probs = test_prob.tolist()
#             else:
#                 test_probs = np.concatenate((test_probs, test_prob.tolist()))
#
#     end_time = time.perf_counter()
#     test_time = end_time - start_time
#     test_labels = np.array(test_labels.tolist())
#     return (
#         cohen_kappa_score(test_preds, test_labels),
#         get_ece(test_probs, test_preds, test_labels),
#         train_time,
#         test_time,
#     )


def prepare_data(
    train_images, train_labels, test_images, test_labels, samples, classes
):

    classes = np.array(list(classes))
    num_classes = len(classes)
    partitions = np.array_split(np.array(range(samples)), num_classes)
    # get indicies of classes we want
    class_idxs = []
    i = 0
    for cls in classes:
        class_idx = np.argwhere(train_labels == cls).flatten()
        np.random.shuffle(class_idx)
        class_idx = class_idx[: len(partitions[i])]
        class_idxs.append(class_idx)
        i += 1

    train_idxs = np.concatenate(class_idxs)
    np.random.shuffle(train_idxs)
    # change the labels to be from 0-len(classes)
    for i in train_idxs:
        train_labels[i] = np.where(classes == train_labels[i])[0][0]

    # get indicies of classes we want
    test_idxs = []
    validation_idxs = []
    for cls in classes:
        test_idx = np.argwhere(test_labels == cls).flatten()
        # out of all, 0.5 validation, 0.5 test
        test_idxs.append(test_idx[int(len(test_idx) * 0.5) :])
        validation_idxs.append(test_idx[: int(len(test_idx) * 0.5)])

    test_idxs = np.concatenate(test_idxs)
    validation_idxs = np.concatenate(validation_idxs)

    # change the labels to be from 0-len(classes)
    for i in test_idxs:
        test_labels[i] = np.where(classes == test_labels[i])[0][0]

    for i in validation_idxs:
        test_labels[i] = np.where(classes == test_labels[i])[0][0]

    train_images = torch.FloatTensor(train_images[train_idxs]).unsqueeze(1)
    train_labels = torch.LongTensor(train_labels[train_idxs])
    valid_images = torch.FloatTensor(test_images[validation_idxs]).unsqueeze(1)
    valid_labels = torch.LongTensor(test_labels[validation_idxs])
    test_images = torch.FloatTensor(test_images[test_idxs]).unsqueeze(1)
    test_labels = torch.LongTensor(test_labels[test_idxs])
    return (
        train_images,
        train_labels,
        valid_images,
        valid_labels,
        test_images,
        test_labels,
    )


# Ax function to initialize the model
def init_net(model, classes, parameters):
    if model == "cnn32":
        net = SimpleCNN32Filter(len(classes))
    elif model == "cnn32_2l":
        net = SimpleCNN32Filter2Layers(len(classes))
    elif model == "cnn32_5l":
        net = SimpleCNN32Filter5Layers(len(classes))
    elif model == "resnet":
        net = models.resnet18(pretrained=True)
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, len(classes))
    return net  # return untrained model


# Add parameters
def training_net(model, parameters, train_data, train_labels):
    # Training loop copied over from run_dn_image_es()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    epochs = parameters.get("epoch", 30)
    batch = 60
    # loss and optimizer
    criterion = nn.CrossEntropyLoss()

    if parameters.get("optimizer", "SGD"):
        optimizer = optim.SGD(
            model.parameters(),
            lr=parameters.get("lr", 0.001),
            momentum=parameters.get("momentum", 0.9),
        )
    if parameters.get("optimizer", "Adam"):
        optimizer = optim.Adam(model.parameters(), lr=parameters.get("lr", 0.001))

    for epoch in range(epochs):  # loop over the dataset multiple times
        model.train()
        for i in range(0, len(train_data), batch):
            # get the inputs
            inputs = train_data[i : i + batch].to(dev)
            labels = train_labels[i : i + batch].to(dev)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model


# Add parameters
def evaluate_net(model, given_data, given_labels, batch, dev):

    # Function that evaluates the model and return the desired metric to optimize. Copied over from run_dn_image_es()

    model.eval()
    prob_cal = nn.Softmax(dim=1)
    test_preds = []
    with torch.no_grad():
        for i in range(0, len(given_data), batch):
            inputs = given_data[i : i + batch].to(dev)
            labels = given_labels[i : i + batch].to(dev)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_preds = np.concatenate((test_preds, predicted.tolist()))

            test_prob = prob_cal(outputs)
            if i == 0:
                test_probs = test_prob.tolist()
            else:
                test_probs = np.concatenate((test_probs, test_prob.tolist()))

    test_labels = np.array(given_labels.tolist())
    return accuracy_score(test_preds, test_labels)


def evaluate_net_final(model, given_data, given_labels, batch, dev):

    # Function that evaluates the cohen kappa.

    model.eval()
    start_time = time.perf_counter()
    prob_cal = nn.Softmax(dim=1)
    test_preds = []
    with torch.no_grad():
        for i in range(0, len(given_data), batch):
            inputs = given_data[i : i + batch].to(dev)
            labels = given_labels[i : i + batch].to(dev)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_preds = np.concatenate((test_preds, predicted.tolist()))

            test_prob = prob_cal(outputs)
            if i == 0:
                test_probs = test_prob.tolist()
            else:
                test_probs = np.concatenate((test_probs, test_prob.tolist()))

    test_labels = np.array(given_labels.tolist())
    end_time = time.perf_counter()
    test_time = end_time - start_time
    return (
        cohen_kappa_score(test_preds, test_labels),
        get_ece(test_probs, test_preds, test_labels),
        test_time,
    )


def run_dn_image_es(
    model,
    train_data,
    train_labels,
    valid_data,
    valid_labels,
    test_data,
    test_labels,
    classes,
):
    """
    Peforms multiclass predictions for a deep network classifier with set number
    of samples and early stopping
    """

    # init_net(parameters, classes)
    # train_evaluate()
    batch = 60

    ax = AxClient(enforce_sequential_optimization=False)

    def train_evaluate(parameterization):
        # Ax primitive that initializes the training sequence --> Trains the model --> Calculates the evaluation metric
        untrained_net = init_net(model, classes, parameterization)
        trained_net = training_net(
            untrained_net, parameterization, train_data, train_labels
        )
        report(
            accuracy=evaluate_net(trained_net, valid_data, valid_labels, batch, device)
        )

    ax.create_experiment(
        name="fsdk18_experiment",
        parameters=[
            {"name": "lr", "type": "range", "bounds": [1e-6, 0.4], "log_scale": True},
            {"name": "momentum", "type": "range", "bounds": [0.0, 1.0]},
            {"name": "epoch", "type": "range", "bounds": [15, 40]},
            {"name": "optimizer", "type": "choice", "values": ["SGD", "Adam"]},
        ],
        objective_name="accuracy",
        minimize=False,
    )

    asha_scheduler = ASHAScheduler(max_t=30, grace_period=5, reduction_factor=2)

    algo = AxSearch(ax_client=ax)
    # Wrap AxSearcher in a concurrently limiter, to ensure that Bayesian optimization receives the
    # data for completed trials before creating more trials
    algo = tune.search.ConcurrencyLimiter(algo, max_concurrent=3)
    tune.run(
        train_evaluate,
        num_samples=20,
        metric="accuracy",
        mode="max",
        search_alg=algo,
        verbose=0,  # Set this level to 1 to see status updates and to 2 to also see trial results.
        scheduler=asha_scheduler,  # To use GPU, specify:
        resources_per_trial={"gpu": 1},
    )

    data = ax.experiment.fetch_data()
    df = data.df
    best_arm_name = df.arm_name[df["mean"] == df["mean"].max()].values[0]
    best_arm = ax.experiment.arms_by_name[best_arm_name]

    best_objectives = np.array(
        [[trial.objective_mean * 100 for trial in ax.experiment.trials.values()]]
    )

    return best_arm, best_objectives
