import operator
import random

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
    train_correct = val_correct = test_correct = 0

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
    for i in range(len(test_set)):
        if posterior_prob(test_set[i], "real") >= 0.5 and test_label[i] == 1:
            test_correct += 1

    # print("validation performance: ", val_correct)
    # print("training performance: ", train_correct)
    print("test performance: ", test_correct)


def posterior_prob(headline, outcome):
    return likelihood_prob(headline, outcome) * prior_prob(outcome)


def get_likehood_table(outcome):
    """
    Get a likelihood table where keys are words, and value is the likelihood probability

    For example, we have total words of 100 for real news
    Trump  13

    likelihood:
    {'Trump': 13/100}
    """
    # training_set, validation_set, test_set, training_label, validation_label, test_label = build_training_validation_test_set()

    real_num = validation_label.count(1)
    fake_num = validation_label.count(0)

    if outcome == "real":
        likelihood_dict_real = {}
        for headline in validation_set:
            for word in headline:
                if not likelihood_dict_real.has_key(word):
                    likelihood_dict_real[word] = 1.
                else:
                    likelihood_dict_real[word] += 1.

        # get likelihoood probabilities
        for key in likelihood_dict_real:
            likelihood_dict_real[key] /= float(real_num)

        # print(likelihood_dict_real)
        return likelihood_dict_real

    if outcome == "fake":
        likelihood_dict_fake = {}
        for headline in validation_set:
            for word in headline:
                if not likelihood_dict_fake.has_key(word):
                    likelihood_dict_fake[word] = 1
                else:
                    likelihood_dict_fake[word] += 1

        for key in likelihood_dict_fake:
            likelihood_dict_fake[key] /= fake_num

        return likelihood_dict_fake


def likelihood_prob(headline, outcome):
    """
    Return the likelihood probability of the headline based on outcome (real or fake)
    """
    likelihood_table = get_likehood_table(outcome)
    for key in likelihood_table:
        if key not in headline:
            likelihood_table[key] = 1. - likelihood_table[key]

    likelihood = 1.
    for key in likelihood_table:
        likelihood *= likelihood_table[key]

    return likelihood


def prior_prob(real_or_fake):
    total_size = len(real_data) + len(fake_data)
    if real_or_fake == "real":
        return len(real_data) / total_size
    if real_or_fake == "fake":
        return len(fake_data) / total_size


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

