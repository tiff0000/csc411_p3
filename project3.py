import operator
import random

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

    dataset = []
    for headline in real_data:
        for word in headline:
            dataset.append([word, 0])

    for headline in fake_data:
        for word in headline:
            dataset.append([word, 1])

    random.shuffle(dataset)
    print(len(dataset))

    validation_index = int(0.7 * len(dataset))
    print(validation_index)
    test_index = int(0.85 * len(dataset))
    print(test_index)

    training_set = [dataset[i][0] for i in range(validation_index)]
    validation_set = [dataset[i][0] for i in range(validation_index, test_index)]
    test_set = [dataset[i][0] for i in range(validation_index, len(dataset) - 1)]

    training_label = [dataset[i][1] for i in range(validation_index)]
    validation_label = [dataset[i][1] for i in range(validation_index, test_index)]
    test_label = [dataset[i][1] for i in range(validation_index, len(dataset) - 1)]

    return training_set, validation_set, test_set, training_label, validation_label, test_label

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

