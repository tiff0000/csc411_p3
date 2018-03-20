import operator
import random
import math
import numpy as np


# from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


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


def build_training_validation_test_set():
    # print(len(real_data) )

    dataset = []
    for headline in real_data:
        line = []
        for word in headline:
            line.append([word, 1])
        dataset.append(line)

    for headline in fake_data:
        line = []
        for word in headline:
            line.append([word, 0])
        dataset.append(line)

    # print(dataset)
    random.seed(4)
    random.shuffle(dataset)

    validation_index = int(0.7 * len(dataset))
    test_index = int(0.85 * len(dataset))

    training_set = []
    for i in range(validation_index):
        headline = []
        for x in range(len(dataset[i])):
            headline.append(dataset[i][x][0])
        training_set.append(headline)

    validation_set = []
    for i in range(validation_index, test_index):
        headline = []
        for x in range(len(dataset[i])):
            headline.append(dataset[i][x][0])
        validation_set.append(headline)

    test_set = []
    for i in range(validation_index, len(dataset) - 1):
        headline = []
        for x in range(len(dataset[i])):
            headline.append(dataset[i][x][0])
        test_set.append(headline)

    # validation_set = [dataset[i][x][0] for i in range(validation_index, test_index) for x in range(len(dataset[i]))]
    # test_set = [[dataset[i][x][0]] for i in range(validation_index, len(dataset) - 1) for x in range(len(dataset[i]))]
    # print(test_set)
    training_label = [dataset[i][0][1] for i in range(validation_index)]
    validation_label = [dataset[i][0][1] for i in range(validation_index, test_index)]
    test_label = [dataset[i][0][1] for i in range(validation_index, len(dataset) - 1)]
    # print(len(test_set))
    # print(len(test_label))

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

    # print(sorted(data.iteritems(), key=lambda (k, v): (v, k))[:])

    frequent_words = []
    for i in range(15):
        highest_frequency_word = max(data.iteritems(), key=operator.itemgetter(1))
        frequent_words.append(highest_frequency_word)
        data.pop(highest_frequency_word[0], None)
    return frequent_words


def part2():
    """
    Get the probability of a headline being fake or real using naive bayes approach
    Steps:
    1. Make a table of word occurrences

    example:
    Word   real  fake
    Trump  0,3   0.4

    """
    # training_set, validation_set, test_set, training_label, validation_label, test_label = build_training_validation_test_set()

    # # validation performance
    # for i in range(len(validation_set)):
    #     if posterior_prob(validation_set[i], "real") >= 0.5 and validation_label[i] == 1:
    #         val_correct += 1
    #
    # # training performance
    # for i in range(len(training_set)):
    #     if posterior_prob(training_set[i], "real") >= 0.5 and training_label[i] == 1:
    #         train_correct += 1

    # test performance

    mI = np.array([1.0, 1e1, 1e2, 1e3, 1e4])
    p_hatI = np.array([0.9, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3])
    countMin = -1.0
    # for m in mI:
    #     for p_hat in p_hatI:
    #         count = likelihood_prob(validation_set, validation_label, m, p_hat)
    #         if count > countMin:
    #             countMin = count
    #             mM = m
    #             p_hatM = p_hat
    likelihood_table = get_likehood_table(1.0, 10)
    accuracy = performance(likelihood_table)
    print("training set performance: " + str(accuracy) + "%")

    # count = likelihood_prob(training_set, training_label, mM, p_hatM)
    # # print("validation performance: ", val_correct)
    # # print("training performance: ", train_correct)
    # print("training set performance: ", count)
    #
    # count = likelihood_prob(test_set, test_label, mM, p_hatM)
    # # print("validation performance: ", val_correct)
    # # print("training performance: ", train_correct)
    # print("test set performance: ", count)


def get_total_word(real_or_fake):
    """
    Return the total number of words in each of the real or fake dataset.
    Input: list of list (list of headlines and sublists of words in the headline)
    """
    total_real_words = 0
    total_fake_words = 0

    for i in range(len(training_set)):
        for word in training_set[i]:
            # increment word occurences for real news
            if training_label[i] == 1:
                total_real_words += 1
            if training_label[i] == 0:
                total_fake_words += 1

    if real_or_fake == "real":
        return total_real_words
    else:
        return total_fake_words


def get_number_of_word_occurrences_table(real_or_fake):
    """
    Return a table (dict) containing the number of occurences for either real of fake news.
    The return structure: a dictionary. key = word, value = number of occurrences of that word
    """

    # make a table which contains the number of occurrences for each word in real and fake
    words_occurrences_real = {}
    words_occurrences_fake = {}
    for i in range(len(training_set)):
        for word in training_set[i]:
            # increment word occurences for real news
            if training_label[i] == 1:
                if not (words_occurrences_real.has_key(word)):
                    words_occurrences_real[word] = 1.
                else:
                    words_occurrences_real[word] += 1.
            if training_label[i] == 0:
                if not (words_occurrences_fake.has_key(word)):
                    words_occurrences_fake[word] = 1.
                else:
                    words_occurrences_fake[word] += 1.

    if real_or_fake == "real":
        return words_occurrences_real
    else:
        return words_occurrences_fake


def get_likehood_table(m, p_hat):
    """
    Get a likelihood table where keys are words, and value is the list of likelihood probabilities for real and fake
    """
    total_real_num = get_total_word("real")
    total_fake_num = get_total_word("fake")

    words_occurrences_real_table = get_number_of_word_occurrences_table("real")
    words_occurrences_fake_table = get_number_of_word_occurrences_table("fake")

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


def predict_headline(likelihood_table, headline):
    """
    Return 1 if predicting the headline as real news, and 0 if predicting the headline as fake news
    """
    likelihood_headline_real = 0
    likelihood_headline_fake = 0

    for word in likelihood_table:
        if word in headline:
            likelihood_headline_real += math.log(likelihood_table[word][0])
            likelihood_headline_fake += math.log(likelihood_table[word][1])
        else:
            likelihood_headline_real += math.log(1 - likelihood_table[word][0])
            likelihood_headline_fake += math.log(1 - likelihood_table[word][1])

    real_prior_prob = get_prior("real")
    fake_prior_prob = get_prior("fake")

    likelihood_headline_real += math.log(real_prior_prob)
    likelihood_headline_fake += math.log(fake_prior_prob)
    # print(likelihood_headline_real, likelihood_headline_fake)

    if likelihood_headline_real >= likelihood_headline_fake:
        print("correct")
        return "real"
    else:
        print("incorrect")
        return "fake"


def get_prior(real_or_fake):
    """
    Return the prior probability of real or fake words
    """
    real_words_num = get_total_word("real")
    fake_words_num = get_total_word("fake")
    total_words = real_words_num + fake_words_num

    if real_or_fake == "real":
        return float(real_words_num) / float(total_words)
    else:
        return float(fake_words_num) / float(total_words)


def performance(likelihood_table):

    correct = 0

    for i in range(len(training_set)):
        headline = training_set[i]
        if (predict_headline(likelihood_table, headline) == "real" and training_label[i] == 1) or\
            (predict_headline(likelihood_table, headline) == "fake" and training_label[i] == 0):
            correct += 1
    print(correct)
    print(len(training_set))
    return (float(correct) / len(training_set)) * 100



if __name__ == "__main__":
    # print top 10 occurences from each real and fake news
    # part1()
    part2()
    # part3()
    # part4()
    # get_likelikehood_table("real")
    # print(get_likehood_table("real"))
    # print(likelihood_prob(training_set[0], "real"))
    # print get_word_num("real")
    # print get_word_num("fake")

    # likelihood_table = get_likehood_table(1.0, 0.9)
    #trump warns of vote flipping on machines
    #donald trump do you remember the year since he was elected
    # result = predict_headline(likelihood_table, ['trump', 'warns', 'of', 'vote', 'flipping', 'on', 'machine'], False)
    # print(result)
    # result = predict_headline(likelihood_table, ['trump', 'warns', 'of', 'vote', 'flipping', 'on', 'machine'], True)
    # print(result)
    # result = predict_headline(likelihood_table, ['donal', 'trump', 'do', 'you', 'remember', 'the', 'year', 'since', 'he', 'was', 'elected'], True)
    # print(result)
    # print(get_prior("real"))
    # print(get_prior("fake"))

    # performance(likelihood_table)

