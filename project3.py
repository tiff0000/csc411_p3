import operator


def part1():
    """
    Print top 10 occurences from each of real and fake data set.
    Get training, validation, and test set
    """
    real_data = get_data_as_list("clean_real.txt")
    fake_data = get_data_as_list("clean_fake.txt")

    # print(real_data)
    # print(fake_data)

    top_10_real = get_frequent_occurences(real_data)
    top_10_fake = get_frequent_occurences(fake_data)

    print ("real: " + str(top_10_real))
    print("fake: " + str(top_10_fake))

    training_set, validation_set, test_set = [], [], []
    training_label, validation_label, test_label = [], [], []

    real_data_size = len(real_data)
    validation_index = int(0.70 * real_data_size)
    test_index = int(0.85 * real_data_size)

    real_data_random = get_word_stats()
    i = 0
    while (i < real_data_size):
        for word in real_data[i]:
            training_set.append(word)
            training_label.append()






def get_data_as_list(filename):
    """
    Read from file and return a list containing all headlines from file
    """
    data = open(filename, "r")
    result = []
    for line in data:
        result.append(line[:-1].split())
    return result


def get_word_stats(input):
    data = {}
    for lines in input:
        for word in lines:
            if data.has_key(word):
                data[word] += 1
            else:
                data[word] = 1
    return data

def get_frequent_occurences(input):
    """
    Get top 10 word occurences from input.
    """
    data = get_word_stats(input)

    print(sorted(data.iteritems(), key=lambda (k, v): (v, k))[:])

    frequent_words = []
    for i in range(15):
        highest_frequency_word = max(data.iteritems(), key=operator.itemgetter(1))
        frequent_words.append(highest_frequency_word)
        data.pop(highest_frequency_word[0], None)
    return frequent_words




if __name__ == "__main__":
    # print top 10 occurences from each real and fake news
    part1()

