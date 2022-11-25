import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
# import sys
# import matplotlib.pyplot as plt
# from tqdm import tqdm

ALPHABET_LEN = 26
XI_LEN = 8256  # n pixels + all pixel pairs
BIAS = 1  # multi class classifier bias

alphabet = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h", 8: "i", 9: "j", 10: "k",
            11: "l", 12: "m", 13: "n", 14: "o", 15: "p", 16: "q", 17: "r", 18: "s", 19: "t", 20: "u",
            21: "v", 22: "w", 23: "x", 24: "y", 25: "z"}

# all 20 classified sequences with lengths equal to the index
sequences = [[],
             [],
             ["bo", "ty"],
             ["max"],
             ["cruz", "drew", "greg", "hugh", "jack"],
             ["brock", "devyn", "elvis", "floyd", "quinn", "ralph", "steve", "tariq"],
             ["dwight", "joseph", "philip"],
             [],
             ["clifford"]]


def get_keys_from_value(my_dict, val):
    for key, val_i in my_dict.items():
        if val_i == val:
            return key
    print("Error, no key found!")
    return None


# load a single image
def load_image(img_path):
    space_idx = img_path.rfind('_')
    Y = img_path[space_idx + 1:-4]
    # n = img_path[space_idx-4:space_idx]
    # print(n, Y)

    img = Image.open(img_path)
    img_mat = np.asarray(img)

    n_letters = len(Y)
    im_height = int(img_mat.shape[0])
    im_width = int(img_mat.shape[1] / n_letters)
    n_pixels = im_height * im_width

    X = np.zeros([int(n_pixels + (n_pixels - 1) * n_pixels / 2), n_letters])

    # compute features of each letter
    for i in range(n_letters):
        # add pixel values in the feature array
        letter = img_mat[:, im_width * i:im_width * (i + 1)] / 255
        X[0:n_pixels, i] = letter.flatten()

        # add the multiples of each pair of the pixels in the f. a.
        index = n_pixels
        for j in range(0, n_pixels - 1):
            for k in range(j + 1, n_pixels):
                X[index, i] = X[j, i] * X[k, i]
                index += 1

        X[:, i] /= np.linalg.norm(X[:, i])

    return X, Y, img


# load all image from a folder
def load_images(img_folder):
    X = []
    Y = []
    img = []

    for file in listdir(img_folder):
        path = join(img_folder, file)
        if isfile(path):
            X_i, Y_i, img_i = load_image(path)
            X.append(X_i)
            Y.append(Y_i)
            img.append(img_i)

    print("Data loaded:", img_folder)
    return X, Y, img

trn_X, trn_Y, trn_img = load_images('ocr_names_images/trn')
test_X, test_Y, test_img = load_images('ocr_names_images/tst')

# ---------------------------------------------------------------------------------------------------------


# returns na array P = [0 ... 0, x_i, 0 ... 0]
def phi(x_i, y_idx):
    phi_xy = np.zeros((ALPHABET_LEN * XI_LEN, 1))
    phi_xy[y_idx * XI_LEN: (y_idx + 1) * XI_LEN] = x_i

    return phi_xy


# with biases added at the end of x_i
# returns na array P = [0 ... 0, x_i, 1, 0 ... 0]
def phi_b(x_i, y_idx):
    phi_xy = np.zeros((ALPHABET_LEN * (XI_LEN + 1), 1))

    phi_xy[y_idx * (XI_LEN + 1): (y_idx + 1) * (XI_LEN + 1) - 1] = x_i
    phi_xy[(y_idx + 1) * (XI_LEN + 1) - 1] = 1

    return phi_xy


# get prediction values for one letter from features x_i
def letter_predictor(w, x_i):
    predictions_list = np.zeros(ALPHABET_LEN)
    for i in range(ALPHABET_LEN):
        y_pred = (w.T @ phi_b(x_i, i))[0][0] if BIAS else (w.T @ phi(x_i, i))[0][0]
        predictions_list[i] = y_pred
    return predictions_list


# prediction of the letter (y_pred) from image(x_i) and weights (w) with weights update
def class_classifier(y_ref, w, x_i, word_error):

    predictions_list = letter_predictor(w, x_i)
    y_pred_idx = np.argmax(predictions_list)

    y_pred = alphabet[y_pred_idx]
    y_ref_idx = get_keys_from_value(alphabet, y_ref)

    if y_pred != y_ref:
        w += phi_b(x_i, y_ref_idx) if BIAS else phi(x_i, y_ref_idx)
        w -= phi_b(x_i, y_pred_idx) if BIAS else phi(x_i, y_pred_idx)
        word_error += 1
    return y_pred, w, word_error


# prediction of the sequence (Y) from image sequence(X) and weights (w) with weights update
# for each letter independently
def multi_class_classifier(Y, w, X):
    word_error = 0
    for j in range(len(Y)):
        y_ref = Y[j]
        x_i = np.vstack(X[:, j])
        y_pred, w, word_error = class_classifier(y_ref, w, x_i, word_error)
    return w, word_error


# learn parameters
def train_weights_1(train_X, train_Y):
    train_dataset_len = 1000

    # weights = [w_1, w_2 ... w_26]
    if BIAS:
        print("Training weights with biases")
        weights = np.zeros((ALPHABET_LEN * (XI_LEN + 1), 1))  # with biases
    else:
        print("Training weights without biases")
        weights = np.zeros((ALPHABET_LEN * XI_LEN, 1))

    n_correct_words = 0
    n_iterations = 0
    min_success_rate = 1000
    while n_correct_words < min_success_rate:
        n_correct_words = 0
        for i in range(train_dataset_len):
            X = train_X[i]
            Y = train_Y[i]
            weights, word_error = multi_class_classifier(Y, weights, X)
            if word_error == 0:
                n_correct_words += 1
        print("Success rate of %d: %d" % (n_iterations, n_correct_words))
        n_iterations += 1

    print("N of iterations for success rate %d: %d" % (min_success_rate, n_iterations))
    return weights


def evaluate_model_1(test_X, test_Y, weights):
    test_dataset_len = 500
    letter_scores = np.zeros(ALPHABET_LEN)
    letter_occurences = np.zeros(ALPHABET_LEN)
    n_correct_words = 0
    for i in range(test_dataset_len):
        X = test_X[i]
        Y = test_Y[i]
        word_error = 0
        for i in range(len(Y)):
            y_ref = Y[i]
            y_ref_idx = get_keys_from_value(alphabet, y_ref)
            letter_occurences[y_ref_idx] += 1
            x_i = np.vstack(X[:, i])
            y_pred_idx = letter_predictor(weights, x_i)
            y_pred = alphabet[y_pred_idx]
            if y_pred == y_ref:
                letter_scores[y_pred_idx] += 1
            else:
                word_error += 1
        if word_error == 0:
            n_correct_words += 1
    # print(letter_scores / letter_occurences)
    R_char = 1 - sum(letter_scores / letter_occurences) / ALPHABET_LEN
    R_seq = 1 - n_correct_words / test_dataset_len
    print("R_char is:", R_char)
    print("R_seq is:", R_seq)
    return


# weights = train_weights_1(trn_X, trn_Y)
# evaluate_model_1(test_X, test_Y, weights)
# print("Number of non zero elements in weights is", np.count_nonzero(weights), "/", weights.shape[0])

# ---------------------------------------------------------------------------------------------------------


# # prediction of the sequence
# def structured_classifier_pairs(Y, w, g, X, word_error, index):
#     x_i = np.vstack(X[:, index])
#     y_ref = Y[index]
#     new_F_list = np.zeros(ALPHABET_LEN)
#
#     # get a prediction list (q_list) as a base value for the new_F_list
#     # and update weights if the argmax val is a wrong estimate
#     q_list = letter_predictor(w, x_i)
#
#     y_ref = Y[index]
#     y_ref_idx = get_keys_from_value(alphabet, y_ref)
#
#     # create the new list of F values -> new_F_list
#     if index == 0:
#         new_F_list = q_list
#
#         y_pred_idx = np.argmax(new_F_list)
#         y_pred = alphabet[y_pred_idx]
#
#         if y_pred != y_ref:
#             w += phi_b(x_i, y_ref_idx) if BIAS else phi(x_i, y_ref_idx)
#             w -= phi_b(x_i, y_pred_idx) if BIAS else phi(x_i, y_pred_idx)
#             word_error += 1
#     else:
#         w, g, F_list, word_error = structured_classifier_pairs(Y, w, g, X, word_error, index-1)
#
#         # fill in values for actual list of F values -> new_F_list
#         for i in range(ALPHABET_LEN):
#             new_F_list[i] = q_list[i] + max(F_list + g[:, i])
#         y_pred_idx = np.argmax(new_F_list - q_list)
#         y_pred = alphabet[y_pred_idx]
#
#         y_ref_previous = Y[index-1]
#         y_ref_idx_previous = get_keys_from_value(alphabet, y_ref_previous)
#
#         # update weights and g if wrong prediction
#
#         if y_pred != y_ref:
#             g[y_ref_idx_previous, y_ref_idx] += 2
#             g[:, y_ref_idx] -= 1
#             w += phi_b(x_i, y_ref_idx) if BIAS else phi(x_i, y_ref_idx)
#             w -= phi_b(x_i, y_pred_idx) if BIAS else phi(x_i, y_pred_idx)
#             word_error += 1
#
#     return w, g, new_F_list, word_error

def find_next_y(Y, w, g, X, F_list, y_seq, index):

    q_list = letter_predictor(w, np.vstack(X[:, index]))

    if index == len(Y)-1:
        # fill in values for actual list of F values -> new_F_list
        new_F_list = np.zeros(ALPHABET_LEN)
        for i in range(ALPHABET_LEN):
            new_F_list[i] = q_list + max(F_list + g[:, i])
        y_seq[index] = np.argmax(new_F_list)
    else:
        # fill in values for actual list of F values -> new_F_list
        new_F_list = np.zeros(ALPHABET_LEN)
        for i in range(ALPHABET_LEN):
            new_F_list[i] = max(F_list + g[:, i])
        y_pred = np.argmax(new_F_list)
        new_F_list += q_list
        y_seq, w, g = find_next_y(Y, w, g, X, F_list, y_seq, index+1)

    y_seq[index-1] = np.argmax(F_list + g[:, i])

    return y_seq

def structured_classifier_pairs(Y, w, g, X):
    index = 0
    y_seq = np.zeros(len(Y))
    q_list = letter_predictor(w, np.vstack(X[:, index]))
    y_seq = find_next_y(Y, w, g, X, q_list, y_seq, index+1)
    return y_seq


# learn parameters
def train_weights_2(train_X, train_Y):
    train_dataset_len = 1000

    # weights = [w_1, w_2 ... w_26]
    weights = np.zeros((ALPHABET_LEN * (XI_LEN + 1), 1))  # with biases

    # letter transition matrix = [g_1; g_2; ... g_26]
    # g[i,j] .. value of probable transition from letter i to the following letter j
    g = np.zeros((ALPHABET_LEN, ALPHABET_LEN))

    n_correct_words = 0
    n_iterations = 0
    min_success_rate = 1000
    while n_correct_words < min_success_rate:
        n_correct_words = 0
        for i in range(train_dataset_len):
            X = train_X[i]
            Y = train_Y[i]
            word_error = 0
            Q = np.zeros(ALPHABET_LEN, len(Y))
            for j in range(len(Y)):
                x_i = np.vstack(X[:, j])
                Q[:, j] = letter_predictor(weights, x_i)

            y_1, weights, g, word_error = structured_classifier_pairs(Q, weights, g, X)
            if word_error == 0:
                n_correct_words += 1
        print("Success rate of %d: %d" % (n_iterations, n_correct_words))
        n_iterations += 1

    print("N of iterations for success rate %d: %d" % (min_success_rate, n_iterations))
    return weights, g


def evaluate_model_2(test_X, test_Y, weights, g):
    test_dataset_len = 500
    letter_scores = np.zeros(ALPHABET_LEN)
    letter_occurences = np.zeros(ALPHABET_LEN)
    n_correct_words = 0
    for i in range(test_dataset_len):
        X = test_X[i]
        Y = test_Y[i]
        word_error = 0
        for i in range(len(Y)):
            y_ref = Y[i]
            y_ref_idx = get_keys_from_value(alphabet, y_ref)
            letter_occurences[y_ref_idx] += 1
            x_i = np.vstack(X[:, i])
            y_pred_idx = letter_predictor(weights, x_i)
            y_pred = alphabet[y_pred_idx]
            if y_pred == y_ref:
                letter_scores[y_pred_idx] += 1
            else:
                word_error += 1
        if word_error == 0:
            n_correct_words += 1
    # print(letter_scores / letter_occurences)
    R_char = 1 - sum(letter_scores / letter_occurences) / ALPHABET_LEN
    R_seq = 1 - n_correct_words / test_dataset_len
    print("R_char is:", R_char)
    print("R_seq is:", R_seq)
    return


weights, g = train_weights_2(trn_X, trn_Y)
evaluate_model_2(test_X, test_Y, weights, g)

# ---------------------------------------------------------------------------------------------------------


# def lsc2_predictor(w, x_i):
#     predictions_list = np.zeros(ALPHABET_LEN)
#     y_idx = -1
#     y_max = -10
#     for i in range(ALPHABET_LEN):
#         if BIAS:
#             y_pred = (w.T @ phi_b(x_i, i))[0][0]
#         else:
#             y_pred = (w.T @ phi(x_i, i))[0][0]
#
#         predictions_list[i] = y_pred
#         if y_pred > y_max:
#             y_max = y_pred
#             y_idx = i
#
#     return y_idx
#
#
# # prediction of the sequence
# def structured_classifier_fixed(y_ref, w, x_i, word_error):
#     y_pred_idx = letter_predictor(w, x_i)
#     y_pred = alphabet[y_pred_idx]
#     y_ref_idx = get_keys_from_value(alphabet, y_ref)
#
#     if y_pred == y_ref:
#         # print("Correct prediction:", y_pred)
#         pass
#     else:
#         # print("INCORRECT prediction:", y_pred, "is not", y_ref)
#         if BIAS:
#             w += phi_b(x_i, y_ref_idx)
#             w -= phi_b(x_i, y_pred_idx)
#         else:
#             w += phi(x_i, y_ref_idx)
#             w -= phi(x_i, y_pred_idx)
#         word_error += 1
#     return w, word_error
#
#
# # learn parameters
# def train_weights_3(train_X, train_Y):
#     train_dataset_len = 1000
#
#     # weights = [w_1, w_2 ... w_26]
#     if BIAS:
#         print("Training weights with biases")
#         weights = np.zeros((ALPHABET_LEN * (XI_LEN + 1), 1))  # with biases
#     else:
#         print("Training weights without biases")
#         weights = np.zeros((ALPHABET_LEN * XI_LEN, 1))
#
#     n_correct_words = 0
#     n_iterations = 0
#     min_success_rate = 1000
#     while n_correct_words < min_success_rate:
#         n_correct_words = 0
#         for i in range(train_dataset_len):
#             X = train_X[i]
#             Y = train_Y[i]
#             weights, word_error = multi_class_classifier(Y, weights, X)
#             if word_error == 0:
#                 n_correct_words += 1
#         print("Success rate of %d: %d" % (n_iterations, n_correct_words))
#         n_iterations += 1
#
#     print("N of iterations for success rate %d: %d" % (min_success_rate, n_iterations))
#     return weights
#
#
# def evaluate_model_3(test_X, test_Y, weights):
#     test_dataset_len = 500
#     letter_scores = np.zeros(ALPHABET_LEN)
#     letter_occurences = np.zeros(ALPHABET_LEN)
#     n_correct_words = 0
#     for i in range(test_dataset_len):
#         X = test_X[i]
#         Y = test_Y[i]
#         word_error = 0
#         for i in range(len(Y)):
#             y_ref = Y[i]
#             y_ref_idx = get_keys_from_value(alphabet, y_ref)
#             letter_occurences[y_ref_idx] += 1
#             x_i = np.vstack(X[:, i])
#             y_pred_idx = letter_predictor(weights, x_i)
#             y_pred = alphabet[y_pred_idx]
#             if y_pred == y_ref:
#                 letter_scores[y_pred_idx] += 1
#             else:
#                 word_error += 1
#         if word_error == 0:
#             n_correct_words += 1
#     # print(letter_scores / letter_occurences)
#     R_char = 1 - sum(letter_scores / letter_occurences) / ALPHABET_LEN
#     R_seq = 1 - n_correct_words / test_dataset_len
#     print("R_char is:", R_char)
#     print("R_seq is:", R_seq)
#     return
#
#
# weights = train_weights_2(trn_X, trn_Y)
# evaluate_model_2(test_X, test_Y, weights)

# ---------------------------------------------------------------------------------------------------------
