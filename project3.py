import operator
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import graphviz



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
    print("Dataset description: ")
    print("real news headlines: " + str(len(real_data)))
    print("fake news headlines: " + str(len(fake_data)))

    top_real = get_frequent_occurences(real_data)
    top_fake = get_frequent_occurences(fake_data)

    least_real = get_frequent_occurences(real_data, True)
    least_fake = get_frequent_occurences(fake_data, True)

    print("top real and least fake")
    words = []
    for key in top_real:
        if least_fake.has_key(key):
            words.append([key, top_real[key], least_fake[key]])
    print(words)

    print("top fake and least real")
    words = []
    for key in top_fake:
        if least_real.has_key(key):
            words.append([key, top_fake[key], least_real[key]])
    print(words)

    # words = []
    # for word_real in top_20_real:
    #     for word_fake in least_20_fake:
    #         if word_real[0] == word_fake[0]:
    #             words.append(word_real)
    # print("top 20 real and least 20 fake")
    # print(words)
    #
    # words = []
    # for word_fake in top_20_fake:
    #     for word_real in least_20_real:
    #         if word_real[0] == word_fake[0]:
    #             words.append(word_real)
    # print("top 20 fake and least 20 real")
    # print(words)

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


def get_frequent_occurences(input, low=False):
    """
    Get top 10 word occurences from input.
    """
    data = get_word_stats(input)
    frequent_words = {}
    not_frequent_words = {}

    for i in range(500):
        if low:
            lowest_frequency_words = min(data.iteritems(), key=operator.itemgetter(1))
            not_frequent_words[lowest_frequency_words[0]] = lowest_frequency_words[1]
            data.pop(lowest_frequency_words[0], None)
        else:
            highest_frequency_word = max(data.iteritems(), key=operator.itemgetter(1))
            frequent_words[highest_frequency_word[0]] = highest_frequency_word[1]
            data.pop(highest_frequency_word[0], None)

    if low:
        return not_frequent_words
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
            likelihood_table = get_probability_table(m, p_hat, validation_set, validation_label)
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

    likelihood_table = get_probability_table(m, p_hat, training_set, training_label)
    accuracy = performance(likelihood_table, training_set, training_label)
    print("training set performance: " + str(accuracy) + "%")

    likelihood_table = get_probability_table(m, p_hat, validation_set, validation_label)
    accuracy = performance(likelihood_table, validation_set, validation_label)
    print("validation set performance: " + str(accuracy) + "%")

    likelihood_table = get_probability_table(m, p_hat, test_set, test_label)
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
                    words_occurrences_real[word] = 1
                else:
                    words_occurrences_real[word] += 1
            if dataset_label[i] == 0:
                if not (words_occurrences_fake.has_key(word)):
                    words_occurrences_fake[word] = 1
                else:
                    words_occurrences_fake[word] += 1

    if real_or_fake == "real":
        return words_occurrences_real
    else:
        return words_occurrences_fake


def get_probability_table(m, p_hat, dataset, dataset_label):
    """
    Get a probability table where keys are words, and value is the list of probabilities for real and fake

    return dict format:  { 'word1', [probability_real, probability_fake], 'word2', [probability_real, probability_fake]}
    """
    total_real_healines = get_total_headlines("real", dataset, dataset_label)
    total_fake_healines = get_total_headlines("fake", dataset, dataset_label)

    words_occurrences_real_table = get_number_of_word_occurrences_table("real", dataset, dataset_label)
    words_occurrences_fake_table = get_number_of_word_occurrences_table("fake", dataset, dataset_label)

    # get the probability table ( e.g P(word_i | real), P(word_i | fake)) of each words in real and fake
    probability_dict = {}  # { 'word', [likelihood_real, likelihood_fake]}
    for real_word in words_occurrences_real_table:
        probability_dict[real_word] = []
    for fake_word in words_occurrences_fake_table:
        if not probability_dict.has_key(fake_word):
            probability_dict[fake_word] = []

    for word in probability_dict:
        # get real words likelihood
        if word in words_occurrences_real_table:
            real_word_likelihood = float(words_occurrences_real_table[word] + m * p_hat) / float(total_real_healines + m)
        else:
            real_word_likelihood = float(m * p_hat) / float(total_real_healines + m)
        probability_dict[word] = [real_word_likelihood]
        # get fake words likelihood
        if word in words_occurrences_fake_table:
            fake_word_likelihood = float(words_occurrences_fake_table[word] + m * p_hat) / float(total_fake_healines + m)
        else:
            fake_word_likelihood = float(m * p_hat) / float(total_fake_healines + m)
        probability_dict[word].append(fake_word_likelihood)

    return probability_dict


def predict_headline(likelihood_table, headline, dataset, dataset_label, part3=False):
    """
    Return "real" if predicting the headline as real news, and "fake" if predicting the headline as fake news
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

    real_prior_prob = get_prior("real", dataset, dataset_label)
    fake_prior_prob = get_prior("fake", dataset, dataset_label)

    # get posterior probability
    posterior_headline_real = likelihood_headline_real + math.log(real_prior_prob)
    posterior_headline_fake = likelihood_headline_fake + math.log(fake_prior_prob)

    if posterior_headline_real >= posterior_headline_fake:
        return "real"
    else:
        return "fake"


def get_prior(real_or_fake, dataset, dataset_label):
    """
    Return the prior probability of real or fake words
    """
    real_healines_num = get_total_headlines("real", dataset, dataset_label)
    fake_healines_num = get_total_headlines("fake", dataset, dataset_label)

    if real_or_fake == "real":
        return float(real_healines_num) / float(len(dataset_label))
    else:
        return float(fake_healines_num) / float(len(dataset_label))


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

    return (float(correct) / len(dataset)) * 100


# best_parameters = tune_parameters()
# m = best_parameters[0]
# p_hat = best_parameters[1]
m = 0.5
p_hat = 0.01

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

    get_strongly_predict("absence", "real", part3_dataset, part3_dataset_label, ONLY_NON_STOP_WORDS)
    get_strongly_predict("absence", "fake", part3_dataset, part3_dataset_label, ONLY_NON_STOP_WORDS)


# debug 3

def get_strongly_predict(presence_or_absence, real_or_fake, dataset, dataset_label, only_non_stop_words=False):
    words_likelihood_table = get_probability_table(m, p_hat, dataset, dataset_label)

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
            if word in ENGLISH_STOP_WORDS:
                # prediction_based_on_presence.pop(word, None)
                prediction_based_on_presence[word] = -10000000
    presence_strongly_predicts_real = sorted(prediction_based_on_presence.items(), key=operator.itemgetter(1))[-10:]

    print(presence_or_absence + " strongly predicts " + real_or_fake)
    for items in presence_strongly_predicts_real:
        print(items[0], items[1])

    return presence_strongly_predicts_real


def sigmoid(x):
    return 1/(1+np.exp(-x))

def f(x, y, theta, lamb):
    x = np.vstack((np.ones((1, x.shape[1])), x))
    output = sigmoid(np.dot(theta.T,x))
    return -sum(y*np.log(output)+(1-y)*np.log((1-output))) + lamb*np.dot(theta.T,theta)


def df(x, y, theta, lamb):
    x = np.vstack((np.ones((1, x.shape[1])), x))
    output = sigmoid(np.dot(theta.T,x))
    return np.dot(x,(output-y).T)+2*lamb*theta

def h(X, Y, theta):
    X = np.vstack((np.ones((1, X.shape[1])), X))
    h = np.dot(theta.T, X)
    count = 0.0
    for i in range(Y.shape[1]):
        if Y[0,i] == 1 and h[0,i] > 0:
            count += 1
        elif Y[0,i] == 0 and h[0,i] < 0:
            count += 1
    return count / Y.shape[1]

def divide(dataset, label):
    fakeset = []
    realset = []
    for i in range(len(dataset)):
        if label[i]==1:
            realset.append(dataset[i])
        else:
            fakeset.append(dataset[i])
    
    return fakeset, realset

def combineData():
    words = []
    for i in real_data:
        words.extend(i)
    for i in fake_data:
        words.extend(i)
    words = sorted(set(words), key=words.index)
    return words

def transform(dataset):
    word = combineData()
    X = np.zeros((len(words),0))
    for i in dataset:
        init = np.zeros((len(word),1))
        for j in range(len(word)):
            if words[j] in i:
                init[j][0] = 1
        X = np.hstack((X,init))
    return X

# part 4
words = combineData()
fTr,rTr = divide(training_set, training_label)
fV, rV = divide(validation_set, validation_label)
fTe, rTe = divide(test_set, test_label)
trainX = np.hstack((transform(fTr),transform(rTr)))
trainY = np.hstack((np.zeros((1,len(fTr))),np.ones((1,len(rTr)))))
validX = np.hstack((transform(fV),transform(rV)))
validY = np.hstack((np.zeros((1,len(fV))),np.ones((1,len(rV)))))
testX = np.hstack((transform(fTe),transform(rTe)))
testY = np.hstack((np.zeros((1,len(fTe))),np.ones((1,len(rTe)))))

def grad_descent(f, df, x, y, init_t, alpha, lamb):
    epsilon=1e-5
    prev_t = init_t - 10 * epsilon
    t = init_t.copy()
    iter = 0
    max_iter = 3000
    loopNum = []
    outTrain = []
    outValid = []
    outTest = []
    while np.linalg.norm(t - prev_t) > epsilon and iter < max_iter:
        prev_t = t.copy()
        t -= alpha * df(x, y, t, lamb)
        if iter % 30 == 0:
            loopNum.append(iter)
            outTrain.append(h(trainX,trainY,t))
            outValid.append(h(validX,validY,t))
            outTest.append(h(testX,testY,t))
        iter += 1
        
    out = [outTrain, outValid, outTest]
    return t, out, loopNum

def part4():  
    theta0 = np.zeros((len(words)+1,1))
    #tuneP = [3e-4,1e-3,3e-3,1e-2,3e-2,1e-1]
    tuneP = [0.003]
    alpha = 0.1
    minP = -1.0
    validAcc = []
    for tune in tuneP:
        theta, output, lN= grad_descent(f, df, trainX, trainY, theta0, alpha, tune)
        accr = h(validX,validY,theta)
        validAcc.append(accr)
        if accr > minP:
            tunedP = tune
            minP = accr
            thetaP = theta
            
    fig = plt.figure(10)    
    plt.semilogx(tuneP, validAcc)
    plt.ylabel("Validation Set Accuracy")
    plt.xlabel("tuned parameter")
    plt.savefig("part4: tune parameter")
    
    print ("optimal parameter"),tunedP, ("optimal accuracy"), minP
    print ("training set accuracy"), h(trainX, trainY, thetaP)
    print ("validation set accuracy"), h(validX, validY, thetaP)
    print ("test set accuracy"), h(testX, testY, thetaP)
    fig1 = plt.figure(20)
    plt.plot(lN, output[0], label = "Train")
    plt.plot(lN, output[1], label = "Validation")
    plt.plot(lN, output[2], label = "Test")
    plt.ylabel("training accuracy")
    plt.xlabel("iteration")
    plt.title("Part4: Learning curve")
    return thetaP

def part6(theta):
    #6a
    thetaList = theta.flatten().tolist()
    sortedList = sorted(thetaList, reverse = True)
    print ("Top 10 positive thetas\n")
    for i in range(10):
        word = words[thetaList.index(sortedList[i])-1]
        print word, sortedList[i], "\n"
    print ("Top 10 negative thetas\n")
    for i in np.arange(-1,-11,-1):
        word = words[sortedList.index(sortedList[i])-1]
        print word, sortedList[i],"\n"
    #6b
    i = 0
    index = 0
    print ("Top 10 positive theta without stop words:\n")
    while i < 10:
        word = words[thetaList.index(sortedList[index])-1]
        if word not in ENGLISH_STOP_WORDS:
            print word, sortedList[index], "\n"
            i +=1
        index += 1
    print "\n"

    i = 0
    index = -1
    print ("Top 10 negative theta without stop words:\n")
    while i < 10:
        word = words[thetaList.index(sortedList[index])-1]
        if word not in ENGLISH_STOP_WORDS:
            print word, sortedList[index],"\n"
            i +=1
        index -= 1
    print "\n"

def h7(x,y,clf):
    t = clf.predict(x)
    count = 0.0
    for i in range(len(y)):
        if t[i] == y[i]:
            count +=1
    return count/len(t)
    
def part7():
    trainx7 = trainX.T
    trainy7 = trainY.flatten()
    validx7 = validX.T
    validy7 = validY.flatten()
    testx7 = testX.T
    testy7 = testY.flatten()
    
    depths = range(50, 1000, 50)
    train_output = []
    valid_output = []
    test_output = []
    minP = -1.0
    
    for depth in depths:
        clf = tree.DecisionTreeClassifier(max_depth=depth)
        clf = clf.fit(trainx7, trainy7)
        tr7 = h7(trainx7,trainy7,clf)
        va7 = h7(validx7,validy7,clf)
        te7 = h7(testx7,testy7,clf)
        train_output.append(tr7)
        valid_output.append(va7)
        test_output.append(te7)
        
        if va7>minP:
            minP = va7
            depthP = depth
            
    print depthP, minP
        
    fig = plt.figure(20)
    plot1, = plt.plot(depths, train_output, label='training set')
    plot2, = plt.plot(depths, valid_output, label='validation set')
    plot3, = plt.plot(depths, test_output, label='test set')
    
    plt.legend([plot1, plot2, plot3], ['Training', 'Validation', 'Test'])
    plt.title('Depths and performances(part7a)')
    plt.xlabel('Max depth')
    plt.ylabel('Performance')
    plt.show()
        
    dot_data = tree.export_graphviz(clf, max_depth=2, out_file=None, feature_names=words)
    export_graphviz(clf, max_depth=2, out_file=dot_data,                              filled=True, rounded=True,
                            special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    with open("Decision_tree(part7b).png", "wb") as f:
            f.write(graph.create_png())

if __name__ == "__main__":
    # part1()
    # part2()
    # part3a()
    part3b()
    # theta = part4()
    # part5()
    # part6(theta)
    # part7()
    # part8():
    # get_frequent_occurences(real_data)

    # print(len(get_number_of_word_occurrences_table("fake", test_set, test_label)))
    # print(len(get_number_of_word_occurrences_table("real", test_set, test_label)))

    # tune_parameters()
    # dataset_label = []
    # get_strongly_predict("presence", "real", part3_dataset, part3_dataset_label)
