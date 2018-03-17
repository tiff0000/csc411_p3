import operator


def part1():
    """
    Print top 10 occurences from each of real and fake data set.
    """

    real_data = get_data_as_list("clean_real.txt")
    fake_data = get_data_as_list("clean_fake.txt")

    top_10_real = get_frequent_occurences(real_data)
    top_10_fake = get_frequent_occurences(fake_data)

    print()
    # print ("real: " + str(top_10_real))
    # print("fake: " + str(top_10_fake))



def get_data_as_list(filename):
    """
    Read from file and return a list containing all headlines from file
    """
    data = open(filename, "r")
    result = []
    for line in data:
        result.append(line[:-1].split())
    return result


def get_frequent_occurences(input):
    """
    Get top 10 word occurences from input.
    """
    data = {}
    for lines in input:
        for word in lines:
            if data.has_key(word):
                data[word] += 1
            else:
                data[word] = 1

    # print(sorted(data.iteritems(), key=lambda (k, v): (v, k))[-10:])

    frequent_words = []
    for i in range(15):
        highest_frequency_word = max(data.iteritems(), key=operator.itemgetter(1))
        frequent_words.append(highest_frequency_word)
        data.pop(highest_frequency_word[0], None)
    return frequent_words


if __name__ == "__main__":
    # print top 10 occurences from each real and fake news
    part1()

