import os
import codecs
import random

import nltk                         # Natural Language Toolkit
from nltk import word_tokenize      # word tokenizer
nltk.download('punkt')              # download the punkt tokenizer

from nltk import NaiveBayesClassifier, classify
from nltk.text import Text


# **** function to read the contents of the files ****
def read_in(folder):
    a_list = []

    # **** loop through the files in the folder ****
    for filename in os.listdir(folder):

        # **** skip hidden files ****
        if filename.startswith('.'):
            continue

        # ???? ????
        #print(' filename: ', filename)

        # **** read in the contents of this file ****
        with codecs.open(os.path.join(folder, filename), 'r', 'utf-8') as f:
            a_list.append(f.read())
    
    return a_list


# **** function to tokenize text ****
def tokenize(text):
    tokens = []

    # **** loop through the words tokenizing text ****
    for word in word_tokenize(text):
        tokens.append(word)

    return tokens


# **** function to extract features ****
def get_features(email):
    features = {}

    # **** ****
    words = tokenize(email)

    # **** loop through the words ****
    for word in words:
        features['contains({})'.format(word.lower())] = True

    return features


# **** function to train a Naive Bayes classifier ****
def train(features, proportion):
    train_size = int(len(features) * proportion)
    train_set, test_set = features[:train_size], features[train_size:]

    # ????? ????
    print(f"Training set size: {len(train_set)} emails")
    print(f"Test set size: {len(test_set)} emails")

    # **** ****
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    return train_set, test_set, classifier


# **** function to evaluate the classifier ****
def evaluate(train_set, test_set, classifier):
    print(f"Accuracy on the training set: {classify.accuracy(classifier, train_set):.3f}")
    print(f"Accuracy of the test set: {classify.accuracy(classifier, test_set):.3f}")
    classifier.show_most_informative_features(12)


# **** concordance function ****
def concordance(data_list, search_word):
    for email in data_list:
        word_list = [word for word in word_tokenize(email.lower())]
        text_list = Text(word_list)
        if search_word in text_list:
            text_list.concordance(search_word)


# **** first step - read the contents of the emails in the spam and ham folders ****

# **** read in spam list ****
spam_list = read_in("C:/Users/johnc/workspace19/Getting-Started-with-NLP/enron1/enron1/spam/")

# **** read in ham list ****
ham_list = read_in("C:/Users/johnc/workspace19/Getting-Started-with-NLP/enron1/enron1/ham/")

# **** display the number of spam emails ****
print('spam_list: ', len(spam_list))

# **** display the number of ham emails ****
print(' ham_list: ', len(ham_list))

# **** display the first spam email ****
print('\nspam_list[0]: ', spam_list[0])

# **** display the first ham email ****
print('\nham_list[0]: ', ham_list[0])


# **** second step - split the email text into words ****

# **** combine the spam and ham lists ****
all_emails = [('spam', email) for email in spam_list]
all_emails += [('ham', email) for email in ham_list]

# **** seed random number generator ****
random.seed(42)

# **** random shuffle the emails ****
random.shuffle(all_emails)

# **** display the number of all_emails ****
print('all_emails: ', len(all_emails))

# **** split the email text into words ****
text = "What's the best way to split a sentence into words in Python?"
print('tokenized text:', tokenize(text))


# **** third step - extract and normalize the features ****
print('\nextract features:')
all_features = [(get_features(email), label) for (label, email) in all_emails]


# ???? ????
print(get_features("Participate In Our New Lottery NOW!"))
print('len(all_features):', len(all_features))
print('len(all_features[0][0]):', len(all_features[0][0]))
print('len(all_features[99][0]):', len(all_features[99][0]))


# **** fouth step - train the classifier ****
print('\ntrain the classifier:')
train_set, test_set, classifier = train(all_features, 0.8)


# **** fifth step - evaluate the classifier ****
print('\nevaluate the classifier:')
evaluate(train_set, test_set, classifier)


# **** ****
print('\n\nSTOCKS in HAM:')
concordance(ham_list, 'stocks')

print('\n\nSTOCKS in SPAM:')
concordance(spam_list, 'stocks')


# **** apply spam filter to new emails ****
print('\napply spam filter to new emails:')

test_spam_list = ['Participate in our new lottery!', "Try out this new medicine"]
test_ham_list = ['See the minutes from the last meeting attached', "Investors are comming to our office on Monday"]

test_emails = [(email_content, 'spam') for email_content in test_spam_list]
test_emails += [(email_content, 'ham') for email_content in test_ham_list]

new_test_set = [(get_features(email), label) for (email, label) in test_emails]

evaluate(train_set, new_test_set, classifier)


# **** print predicted label ****
print('\nprint predicted label:\n')
for email in test_spam_list:
    print(f"email: {email}")
    print(f"Predicted label: {classifier.classify(get_features(email))}")

for email in test_ham_list:
    print(f"email: {email}")
    print(f"Predicted label: {classifier.classify(get_features(email))}")


# **** classify email read from console ****
print('\nclassify email read from console:\n')
while True:
    email = input("Enter an email (<Return> to exit): ")
    if len(email) == 0:
        break
    print(f"Predicted label: {classifier.classify(get_features(email))}")
