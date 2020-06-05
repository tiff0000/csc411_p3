# machine-learning-recognize-fake-news
A total of 1298 “fake news” headlines (which mostly include headlines of articles classified as biased etc.) and 1968 “real” news headlines, where the “fake news” headlines are from https://www.kaggle.com/mrisdal/fake-news/data and “real news” headlines are from https://www.kaggle.com/therohk/million-headlines. We further cleaned the data by removing words from fake news titles that are not a part of the headline, removing special characters from the headlines, and restricting real news headlines to those after October 2016 containing the word “trump”. For your interest, the cleaning script is available at clean_script.py, but you do not need to run it. The cleaned-up data is available below:
Each headline appears as a single line in the data file. Words in the headline are separated by spaces, so just use str.split() in Python to split the headlines into words.

Real news headlines: clean_real.txt
Fake news headlines: clean_fake.txt

content:
Part1: Predicting whether a headline is real or fake news from words that appear in the headline.
3 examples of specific keywords that may be useful, together with statistics on how often they appear in real and fake headlines.

Part2: Implement the Naive Bayes algorithm for predicting whether a headline is real or fake. Tune the parameters of the prior (called 
m and p using the validation set.

Part3: 
List 10 words whose presence most strongly predicts that the news is real.

List 10 words whose absence most strongly predicts that the news is real.

List 10 words whose presence most strongly predicts that the news is fake.

List 10 words whose absence most strongly predicts that the news is fake.

Part4:
list 10 non-stopwords that most strongly predict that the news is real, and the 10 non-stopwords that most strongly predict that the news is fake.

Part5:
Train a Logistic Regression model on the same dataset. 

Part6:
Display a list of top 10 positive θ and negative θ obtained from Logistic Regression

Part7:
build a decision tree to classify real vs fake news headlines. Used the DecisionTreeClassifier included in sklearn.

Part8:
Computed the mutual information of the split on the training data.

