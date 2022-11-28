import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
# from tqdm import tqdm

ALPHABET_LEN = 26
XI_LEN = 8256  # n pixels + all pixel pairs
BIAS = 1  # multi class classifier bias

alphabet = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h", 8: "i", 9: "j", 10: "k",
            11: "l", 12: "m", 13: "n", 14: "o", 15: "p", 16: "q", 17: "r", 18: "s", 19: "t", 20: "u",
            21: "v", 22: "w", 23: "x", 24: "y", 25: "z"}


def get_keys_from_value(my_dict, val):
    for key, val_i in my_dict.items():
        if val_i == val:
            return key
    print("Error, no key found!")
    return None


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


# prediction of the sequence (Y_pred) from image sequence(X) and weights (w) with weights update
# for each letter independently
def multi_class_classifier(Y_ref, w, X, train=True):
    word_error = 0
    word_len = len(Y_ref)
    Y_pred_idx = np.zeros(len(Y_ref), dtype=int)
    for j in range(word_len):
        x_j = np.vstack(X[:, j])
        predictions_list = letter_predictor(w, x_j)
        Y_pred_idx[j] = np.argmax(predictions_list)

    Y_ref_idx = np.zeros(word_len, dtype=int)
    for i in range(word_len):
        Y_ref_idx[i] = get_keys_from_value(alphabet, Y_ref[i])

    for j in range(word_len):
        if Y_pred_idx[j] != Y_ref_idx[j]:
            if train:
                x_i = np.vstack(X[:, j])
                w += phi_b(x_i, Y_ref_idx[j]) if BIAS else phi(x_i, Y_ref_idx[j])
                w -= phi_b(x_i, Y_pred_idx[j]) if BIAS else phi(x_i, Y_pred_idx[j])
            word_error += 1

    return w, word_error


# learn parameters
def train_weights_1(train_X, train_Y):
    train_dataset_len = 1000

    # weights = [w_1, w_2 ... w_26]
    weights = np.zeros((ALPHABET_LEN * (XI_LEN + 1), 1)) if BIAS else \
                np.zeros((ALPHABET_LEN * XI_LEN, 1))

    n_correct_words = 0
    n_iterations = 0
    min_success_rate = 1000
    while n_correct_words < min_success_rate:
        n_correct_words = 0
        for i in range(train_dataset_len):
            X = train_X[i]
            Y = train_Y[i]
            weights, word_error = multi_class_classifier(Y, weights, X, True)
            if word_error == 0:
                n_correct_words += 1
        print("Success rate of %d: %d" % (n_iterations, n_correct_words))
        n_iterations += 1

    print("N of iterations for success rate %d: %d" % (min_success_rate, n_iterations))
    return weights


def evaluate_model_1(test_X, test_Y, weights):
    n_words = 500
    n_wrong_words = 0
    n_letters = 0
    n_wrong_letters = 0

    for i in range(n_words):
        X = test_X[i]
        Y = test_Y[i]
        weights, word_error = multi_class_classifier(Y, weights, X, False)

        if word_error != 0:
            n_wrong_words += 1
        n_letters += len(Y)
        n_wrong_letters += word_error

    R_char = n_wrong_letters / n_letters
    R_seq = n_wrong_words / n_words

    print("R_char is:", R_char)
    print("R_seq is:", R_seq)
    return


# weights = train_weights_1(trn_X, trn_Y)
# evaluate_model_1(test_X, test_Y, weights)

# ---------------------------------------------------------------------------------------------------------


def find_next_y(Q, w, g, F_list, Y_pred, index):
    q_list = Q[:, index]
    new_F_list = np.zeros(ALPHABET_LEN)
    for i in range(ALPHABET_LEN):
        new_F_list[i] = q_list[i] + max(F_list + g[:, i])

    if index == len(Y_pred)-1:
        Y_pred[index] = np.argmax(new_F_list)
    else:
        Y_pred = find_next_y(Q, w, g, new_F_list, Y_pred, index+1)

    Y_pred[index-1] = np.argmax(F_list + g[:, Y_pred[index]])
    return Y_pred


def structured_classifier_pairs(Y_ref, w, g, X, train=True):
    index = 0

    Q = np.zeros((ALPHABET_LEN, len(Y_ref)))
    for j in range(len(Y_ref)):
        x_i = np.vstack(X[:, j])
        Q[:, j] = letter_predictor(w, x_i)
    q_list = Q[:, index]

    Y_pred_idx = np.zeros(len(Y_ref), dtype=int)
    Y_pred_idx = find_next_y(Q, w, g, q_list, Y_pred_idx, index+1)

    Y_ref_idx = np.zeros(len(Y_ref), dtype=int)
    for i in range(len(Y_ref)):
        Y_ref_idx[i] = get_keys_from_value(alphabet, Y_ref[i])

    word_error = 0
    for j in range(len(Y_ref)):
        if Y_pred_idx[j] != Y_ref_idx[j]:
            word_error += 1
            if train:
                x_i = np.vstack(X[:, j])
                w += phi_b(x_i, Y_ref_idx[j]) if BIAS else phi(x_i, Y_ref_idx[j])
                w -= phi_b(x_i, Y_pred_idx[j]) if BIAS else phi(x_i, Y_pred_idx[j])
        if train and j > 0:
            if Y_pred_idx[j] != Y_ref_idx[j] and Y_pred_idx[j-1] == Y_ref_idx[j-1]:
                g[Y_ref_idx[j-1], Y_pred_idx[j]] -= 1
                g[Y_ref_idx[j-1], Y_ref_idx[j]] += 1
            elif Y_pred_idx[j] == Y_ref_idx[j] and Y_pred_idx[j-1] != Y_ref_idx[j-1]:
                g[Y_pred_idx[j-1], Y_ref_idx[j]] -= 1
                g[Y_ref_idx[j-1], Y_ref_idx[j]] += 1
            elif Y_pred_idx[j] != Y_ref_idx[j] and Y_pred_idx[j-1] != Y_ref_idx[j-1]:
                g[Y_pred_idx[j-1], Y_pred_idx[j]] -= 1
                g[Y_ref_idx[j-1], Y_ref_idx[j]] += 1

    return w, g, word_error


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
            weights, g, word_error = structured_classifier_pairs(Y, weights, g, X, True)

            if word_error == 0:
                n_correct_words += 1
        print("Success rate of %d: %d" % (n_iterations, n_correct_words))
        n_iterations += 1

    print("N of iterations for success rate %d: %d" % (min_success_rate, n_iterations))
    return weights, g


def evaluate_model_2(test_X, test_Y, weights, g):
    n_words = 500
    n_wrong_words = 0
    n_letters = 0
    n_wrong_letters = 0

    for i in range(n_words):
        X = test_X[i]
        Y = test_Y[i]
        weights, g, word_error = structured_classifier_pairs(Y, weights, g, X, False)

        if word_error != 0:
            n_wrong_words += 1
        n_letters += len(Y)
        n_wrong_letters += word_error

    R_char = n_wrong_letters / n_letters
    R_seq = n_wrong_words / n_words

    print("R_char is:", R_char)
    print("R_seq is:", R_seq)
    return


# weights, g = train_weights_2(trn_X, trn_Y)
# evaluate_model_2(test_X, test_Y, weights, g)

# ---------------------------------------------------------------------------------------------------------


# prediction of the sequence
def structured_classifier_fixed(Y_ref, w, X, train=True):
    word_error = 0
    word_len = len(Y_ref)
    sequences_L = sequences[word_len]
    V_seq_param = np.zeros(len(sequences_L))
    Q = np.zeros((ALPHABET_LEN, word_len))
    for j in range(word_len):
        x_i = np.vstack(X[:, j])
        Q[:, j] = letter_predictor(w, x_i)


    Y_pred = np.zeros((word_len))
    max_seq_score = -1000
    max_seq_idx = 0
    for j in range(len(sequences_L)):
        seq = sequences_L[j]
        score = 0
        for k in range(word_len):
            letter_idx = get_keys_from_value(alphabet, seq[k])
            score += Q[letter_idx, k]  # Q[letter_type, word_position]
        score += V_seq_param[j]
        if score > max_seq_score:
            max_seq_score = score
            max_seq_idx = j
    Y_pred = sequences_L[max_seq_idx]

    Y_ref_idx = np.zeros(len(Y_ref), dtype=int)
    Y_pred_idx = np.zeros(len(Y_ref), dtype=int)
    for i in range(word_len):
        Y_ref_idx[i] = get_keys_from_value(alphabet, Y_ref[i])
        Y_pred_idx[i] = get_keys_from_value(alphabet, Y_pred[i])

    for j in range(word_len):
        if Y_pred_idx[j] != Y_ref_idx[j]:
            if train:
                x_i = np.vstack(X[:, j])
                w += phi_b(x_i, Y_ref_idx[j]) if BIAS else phi(x_i, Y_ref_idx[j])
                w -= phi_b(x_i, Y_pred_idx[j]) if BIAS else phi(x_i, Y_pred_idx[j])
            word_error += 1

    if train and word_error > 0:
        v_ref_idx = 0  # correct sequence (from training data) index in sequences_L list
        for j in range(len(sequences_L)):
            seq = sequences_L[j]
            for k in range(word_len):
                letter = seq[k]
                if Y_pred[k] != letter:
                    break
                v_ref_idx = j
        V_seq_param[v_ref_idx] += 1
        V_seq_param[max_seq_idx] -= 1

    return w, word_error


# learn parameters
def train_weights_3(train_X, train_Y):
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
            weights, word_error = structured_classifier_fixed(Y, weights, X, True)

            if word_error == 0:
                n_correct_words += 1
        print("Success rate of %d: %d" % (n_iterations, n_correct_words))
        n_iterations += 1

    print("N of iterations for success rate %d: %d" % (min_success_rate, n_iterations))
    return weights


def evaluate_model_3(test_X, test_Y, weights):
    n_words = 500
    n_wrong_words = 0
    n_letters = 0
    n_wrong_letters = 0

    for i in range(n_words):
        X = test_X[i]
        Y = test_Y[i]
        weights, word_error = structured_classifier_fixed(Y, weights, X, False)

        if word_error != 0:
            n_wrong_words += 1
        n_letters += len(Y)
        n_wrong_letters += word_error

    R_char = n_wrong_letters / n_letters
    R_seq = n_wrong_words / n_words

    print("R_char is:", R_char)
    print("R_seq is:", R_seq)
    return


weights = train_weights_3(trn_X, trn_Y)
evaluate_model_3(test_X, test_Y, weights)

# ---------------------------------------------------------------------------------------------------------
