import operator
import random
import math
import numpy as np



from stop_words import ENGLISH_STOP_WORDS

# print(ENGLISH_STOP_WORDS)

def get_data_as_list(filename):
    """
    Read from file and return a list containing all headlines from file
    """
    data = open(filename, "r")
    result = []
    for line in data:
        result.append(line[:-1].split())
    return result


real_data = get_data_as_list("clean_real.txt")
fake_data = get_data_as_list("clean_fake.txt")


def part1():
    """
    Print top 10 occurences from each of real and fake data set.
    Get training, validation, and test set
    """
    # print(real_data)
    # print(fake_data)

    top_10_real = get_frequent_occurences(real_data)
    top_10_fake = get_frequent_occurences(fake_data)

    # print ("real: " + str(top_10_real))
    # print("fake: " + str(top_10_fake))

    build_training_validation_test_set()

total_dataset = [] # dataset containing headlines of real and fake data

part3_dataset = []
part3_dataset_label = []

def build_training_validation_test_set():

    for headline in real_data:
        line = []
        for word in headline:
            line.append([word, 1])
        total_dataset.append(line)

    for headline in fake_data:
        line = []
        for word in headline:
            line.append([word, 0])
        total_dataset.append(line)

    # part 3  can comment this section out if not running part 3 to get faster answer
    for i in range(len(total_dataset)):
        part3_dataset_label.append(total_dataset[i][0][1])
        headline = []
        for x in range(len(total_dataset[i])):
            headline.append(total_dataset[i][x][0])
        part3_dataset.append(headline)
    # part 3 end -----

    random.seed(4)
    random.shuffle(total_dataset)

    validation_index = int(0.7 * len(total_dataset))
    test_index = int(0.85 * len(total_dataset))

    training_set = []
    for i in range(validation_index):
        headline = []
        for x in range(len(total_dataset[i])):
            headline.append(total_dataset[i][x][0])
        training_set.append(headline)

    validation_set = []
    for i in range(validation_index, test_index):
        headline = []
        for x in range(len(total_dataset[i])):
            headline.append(total_dataset[i][x][0])
        validation_set.append(headline)

    test_set = []
    for i in range(validation_index, len(total_dataset) - 1):
        headline = []
        for x in range(len(total_dataset[i])):
            headline.append(total_dataset[i][x][0])
        test_set.append(headline)

    training_label = [total_dataset[i][0][1] for i in range(validation_index)]
    validation_label = [total_dataset[i][0][1] for i in range(validation_index, test_index)]
    test_label = [total_dataset[i][0][1] for i in range(validation_index, len(total_dataset) - 1)]

    return training_set, validation_set, test_set, training_label, validation_label, test_label


training_set, validation_set, test_set, training_label, validation_label, test_label = build_training_validation_test_set()


def get_word_stats(input):
    """
    Returns a dictionary containing word as key and number of occurrences of the word as value
    """
    data = {}
    for lines in input:
        for word in lines:
            if data.has_key(word):
                data[word] += 1
            else:
                data[word] = 1
    return data


def get_word_num(outcome):
    """
    Return the total number of words in each of the real or fake dataset.
    Input: list of list (list of headlines and sublists of words in the headline)
    """
    if outcome == "real":
        data = real_data
    else:
        data = fake_data
    count = 0
    for headline in data:
        for word in headline:
            count += 1
    return count


def get_frequent_occurences(input):
    """
    Get top 10 word occurences from input.
    """
    data = get_word_stats(input)
    frequent_words = []

    for i in range(15):
        highest_frequency_word = max(data.iteritems(), key=operator.itemgetter(1))
        frequent_words.append(highest_frequency_word)
        data.pop(highest_frequency_word[0], None)
    return frequent_words


def tune_parameters():
    """
    Tune the parameters m and p_hat to get best performance
    """
    m_array = np.array([0.48, 0.49, 0.5, 0.51, 0.52])
    p_hat_array = np.array([0.1])

    best_performance = 0
    best_m = 0
    best_p_hat = 0
    for m in m_array:
        for p_hat in p_hat_array:
            print("m, p_hat = " + str(m) + " " + str(p_hat))
            likelihood_table = get_likehood_table(m, p_hat, validation_set, validation_label)
            current_performance = performance(likelihood_table, validation_set, validation_label)
            if current_performance > best_performance:
                best_m = m
                best_p_hat = p_hat
    return [best_m, best_p_hat]


def part2():
    """
    Use Naive Bayes to evaluate the performance accuracy
    """
    m = 0.5
    p_hat = 0.1
    # m = 0.1
    # p_hat = 10

    likelihood_table = get_likehood_table(m, p_hat, training_set, training_label)
    accuracy = performance(likelihood_table, training_set, training_label)
    print("training set performance: " + str(accuracy) + "%")

    likelihood_table = get_likehood_table(m, p_hat, validation_set, validation_label)
    accuracy = performance(likelihood_table, validation_set, validation_label)
    print("validation set performance: " + str(accuracy) + "%")

    likelihood_table = get_likehood_table(m, p_hat, test_set, test_label)
    accuracy = performance(likelihood_table, test_set, test_label)
    print("test set performance: " + str(accuracy) + "%")


def get_total_headlines(real_or_fake, dataset, dataset_label):
    """
    Return the total number of words in each of the real or fake dataset.
    Input: list of list (list of headlines and sublists of words in the headline)
    """
    total_real_headlines = 0
    total_fake_healines = 0

    for i in range(len(dataset)):
        for word in dataset[i]:
            if dataset_label[i] == 1:
                total_real_headlines += 1
            if dataset_label[i] == 0:
                total_fake_healines += 1

    if real_or_fake == "real":
        return total_real_headlines
    else:
        return total_fake_healines


def get_number_of_word_occurrences_table(real_or_fake, dataset, dataset_label):
    """
    Return a table (dict) containing the number of occurences for either real of fake news.
    The return structure: a dictionary. key = word, value = number of occurrences of that word
    """
    # make a table which contains the number of occurrences for each word in real and fake
    words_occurrences_real = {}
    words_occurrences_fake = {}

    for i in range(len(dataset)):
        headline_without_duplicates = list(set(dataset[i]))
        for word in headline_without_duplicates:
            # increment word occurences for real news
            if dataset_label[i] == 1:
                if not (words_occurrences_real.has_key(word)):
                    words_occurrences_real[word] = 1.
                else:
                    words_occurrences_real[word] += 1.
            if dataset_label[i] == 0:
                if not (words_occurrences_fake.has_key(word)):
                    words_occurrences_fake[word] = 1.
                else:
                    words_occurrences_fake[word] += 1.

    if real_or_fake == "real":
        return words_occurrences_real
    else:
        return words_occurrences_fake


def get_likehood_table(m, p_hat, dataset, dataset_label):
    """
    Get a likelihood table where keys are words, and value is the list of likelihood probabilities for real and fake

    return dict format:  { 'word1', [likelihood_real, likelihood_fake], 'word2', [likelihood_real, likelihood_fake]}
    """
    total_real_num = get_total_headlines("real", dataset, dataset_label)
    total_fake_num = get_total_headlines("fake", dataset, dataset_label)

    words_occurrences_real_table = get_number_of_word_occurrences_table("real", dataset, dataset_label)
    words_occurrences_fake_table = get_number_of_word_occurrences_table("fake", dataset, dataset_label)

    for key in words_occurrences_fake_table:
        if (words_occurrences_fake_table[key] / total_fake_num) >= 1:
            print(words_occurrences_fake_table[key])
            print(total_fake_num)
            print (key, words_occurrences_fake_table[key])

    # get the likelihood table ( e.g P(word_i | real), P(word_i | fake)) of each words in real and fake
    likelihood_dict = {}  # { 'word', [likelihood_real, likelihood_fake]}
    for real_word in words_occurrences_real_table:
        likelihood_dict[real_word] = []
    for fake_word in words_occurrences_fake_table:
        if not likelihood_dict.has_key(fake_word):
            likelihood_dict[fake_word] = []

    for word in likelihood_dict:
        # get real words likelihood
        if word in words_occurrences_real_table:
            real_word_likelihood = float(words_occurrences_real_table[word] + m * p_hat) / float(total_real_num + m)
        else:
            real_word_likelihood = float(m * p_hat) / float(total_real_num + m)
        likelihood_dict[word] = [real_word_likelihood]
        # get fake words likelihood
        if word in words_occurrences_fake_table:
            fake_word_likelihood = float(words_occurrences_fake_table[word] + m * p_hat) / float(total_fake_num + m)
        else:
            fake_word_likelihood = float(m * p_hat) / float(total_fake_num + m)
        likelihood_dict[word].append(fake_word_likelihood)

    return likelihood_dict


def predict_headline(likelihood_table, headline, dataset, dataset_label, part3=False):
    """
    Return "real" if predicting the headline as real news, and "fake" if predicting the headline as fake news
    """
    likelihood_headline_real = 0
    likelihood_headline_fake = 0

    for word in likelihood_table:
        if likelihood_table[word][1] >= 1:
            print(likelihood_table[word][1])
        if word in headline:
            likelihood_headline_real += math.log(likelihood_table[word][0])
            likelihood_headline_fake += math.log(likelihood_table[word][1])
        else:
            likelihood_headline_real += math.log(1 - likelihood_table[word][0])
            likelihood_headline_fake += math.log(1 - likelihood_table[word][1])

    real_prior_prob = get_prior("real", dataset, dataset_label)
    fake_prior_prob = get_prior("fake", dataset, dataset_label)

    # get posterior probability
    posterior_headline_real = likelihood_headline_real + math.log(real_prior_prob)
    posterior_headline_fake = likelihood_headline_fake + math.log(fake_prior_prob)
    # print(likelihood_headline_real, likelihood_headline_fake)

    if posterior_headline_real >= posterior_headline_fake:
        # print("correct")
        return "real"
    else:
        # print("incorrect")
        return "fake"


def get_prior(real_or_fake, dataset, dataset_label):
    """
    Return the prior probability of real or fake words
    """
    real_healines_num = get_total_headlines("real", dataset, dataset_label)
    fake_healines_num = get_total_headlines("fake", dataset, dataset_label)

    total_headlines = real_healines_num + fake_healines_num

    if real_or_fake == "real":
        return float(real_healines_num) / float(total_headlines)
    else:
        return float(fake_healines_num) / float(total_headlines)


def performance(likelihood_table, dataset, dataset_label):
    """
    Get the performance in percentage
    (percentage = number of correct prediction / total headlines * 100)
    """
    correct = 0

    for i in range(len(dataset)):
        headline = dataset[i]
        if (predict_headline(likelihood_table, headline, dataset, dataset_label) == "real" and dataset_label[i] == 1) or\
            (predict_headline(likelihood_table, headline, dataset, dataset_label) == "fake" and dataset_label[i] == 0):
            correct += 1
    # print(correct)
    # print(len(training_set))
    # print(str(float(correct) / len(dataset) * 100) + "%")
    return (float(correct) / len(dataset)) * 100


# best_parameters = tune_parameters()
# m = best_parameters[0]
# p_hat = best_parameters[1]
m = 0.5
p_hat = 0.1

# print("m = " + str(m))
# print("p_hat = " + str(p_hat))


def part3a():
    # presence strongly predicts real
    get_strongly_predict("presence", "real", part3_dataset, part3_dataset_label)

    # absence strongly predicts real
    get_strongly_predict("absence", "real", part3_dataset, part3_dataset_label)

    # presence strongly predicts fake
    get_strongly_predict("presence", "fake", part3_dataset, part3_dataset_label)

    # absence strongly predicts fake
    get_strongly_predict("absence", "fake", part3_dataset, part3_dataset_label)


def part3b():
    ONLY_NON_STOP_WORDS = True
    get_strongly_predict("presence", "real", part3_dataset, part3_dataset_label, ONLY_NON_STOP_WORDS)
    get_strongly_predict("presence", "fake", part3_dataset, part3_dataset_label, ONLY_NON_STOP_WORDS)


# debug 3

def get_strongly_predict(presence_or_absence, real_or_fake, dataset, dataset_label, only_non_stop_words=False):
    words_likelihood_table = get_likehood_table(m, p_hat, dataset, dataset_label)

    # if presence_or_absence == "presence":
        # p(real | presence)

    prediction_based_on_presence = {}

    for word in words_likelihood_table:
        if presence_or_absence == "absence":
            words_likelihood_table[word][0] = 1 - words_likelihood_table[word][0]
            words_likelihood_table[word][1] = 1 - words_likelihood_table[word][1]

        words_likelihood_table[word][0] += get_prior("real", dataset, dataset_label)
        words_likelihood_table[word][1] += get_prior("fake", dataset, dataset_label)

        if real_or_fake == "real":
            how_strong = words_likelihood_table[word][0] - words_likelihood_table[word][1]
            prediction_based_on_presence[word] = how_strong

        if real_or_fake == "fake":
            how_strong = words_likelihood_table[word][1] - words_likelihood_table[word][0]
            prediction_based_on_presence[word] = how_strong

    # only get non-stop words
    if only_non_stop_words:
        for word in prediction_based_on_presence:
            if prediction_based_on_presence[word] in ENGLISH_STOP_WORDS:
                prediction_based_on_presence[word] = -10000000
    presence_strongly_predicts_real = sorted(prediction_based_on_presence.items(), key=operator.itemgetter(1))[-10:]

    print(presence_or_absence + " strongly predicts " + real_or_fake)
    for items in presence_strongly_predicts_real:
        print(items[0], items[1])

    return presence_strongly_predicts_real

# def part4():


if __name__ == "__main__":
    # part1()
    # part2()
    part3a()
    # part3b()
    # part4()
    # part5()
    # part6()
    # part7()
    # part8():
    # print(len(training_set))
    # print(len(training_label))
    # count = 0
    # for i in range(len(training_set)):
    #     for word in training_set[i]:
    #         if word == "trump" and training_label[i] == 0:
    #             count += 1
    # print(count)
    #
    # # print(get_total_headlines("fake", training_set, training_label))
    #
    # total_fake_healines =
    # table = get_likehood_table(m, p_hat, training_set, training_label)
    #
    # for key in table:
    #     if (table[key][1] >= 1):
    #         print(key, table[key])
    # tune_parameters()
    # dataset_label = []
    # get_strongly_predict("presence", "real", part3_dataset, part3_dataset_label)
