#!/usr/bin/env python3
import csv
import sys
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from random import shuffle
from scipy.cluster.vq import whiten
from sklearn.cluster import KMeans
import math
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from nltk.probability import FreqDist

csv.field_size_limit(sys.maxsize)

sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

# get english stopwords
en_stopwords = set(stopwords.words('english'))

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

#Function used for intermediate and final models to apply classifier to data
def PredictAuthors(train, test, authors, stats):
    # Create a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=10)

    # Train the model using the training sets
    train_authors = authors[0:len(train)]
    test_authors = authors[len(train):]
    model.fit(train, train_authors)

    # Predict Output
    predicted = model.predict(test)

    #count if predictions were correct or not
    tp=0
    tn =0
    fn=0
    fp=0
    for i in range(len(predicted)):
        if predicted[i]=="mass":
            if predicted[i] == test_authors[i]:
                tn+=1
            else:
                fn+=1
        else:
            if predicted[i]== test_authors[i]:
                tp+=1
            else:
                fp+=1

    #display overall metrics
    print("tp: {}\tfn: {}\nfp: {}\ttn {}".format(tp, fn, fp, tn))
    print("accuracy: {}".format((tp+tn)/(tp+tn+fn+fp)))
    precision = (tp) / (tp + fp)
    recall = (tp) / ( fn + tp)
    print("when it predicts yes, how often is it correct-- precision: {}".format(precision))
    print("when it's actually yes, how often does it predict yes -- recall: {}".format(recall))
    print("F-score: {}".format(2*precision*recall / (precision + recall)))

    #relay feature importances and feature values back to the user
    importance = model.feature_importances_

    # words per sentence, sentence len variation, lex diversity, unique word count, unique bigram count, commas, exc, question
    names = ["avg words per sentence", "avg sentence length variation", "avg lexical diversity", "avg unique word count",
             "avg unique bigram count", "avg , usage per post", "avg ! usage per post", "avg ? usage per post"]
    print("\nDistinct Authorship Features")
    print("-"*125)
    print("{:35} {:20} {:50} {:20}".format("Feature Name", "Importance in Model", "User Stat", "Mass Stat"))
    print("-" * 125)
    if len(importance) == 8:  #we know we're using all of the features
        print("{:35} {:<20.4f} {:<50.4f} {:<20.4f}".format(names[0], importance[0]*100, stats['author']['avg_words_per_sent'], stats['mass']['avg_words_per_sent']))
        print("{:35} {:<20.4f} {:<50.4f} {:<20.4f}".format(names[1], importance[1] * 100, stats['author']['avg_sentence_length_variation'], stats['mass']['avg_sentence_length_variation']))
        print("{:35} {:<20.4f} {:<50.4f} {:<20.4f}".format(names[2], importance[2] * 100, stats['author']['avg_lex_diversity'] *100,stats['mass']['avg_lex_diversity'] *100))
        print("{:35} {:<20.4f} {:<50.4f} {:<20.4f}".format(names[5], importance[5] * 100, stats['author']['avg_comma'],stats['mass']['avg_comma']))
        print("{:35} {:<20.4f} {:<50.4f} {:<20.4f}".format(names[6], importance[6] * 100, stats['author']['avg_exc'], stats['mass']['avg_exc']))
        print("{:35} {:<20.4f} {:<50.4f} {:<20.4f}".format(names[7], importance[7] * 100, stats['author']['avg_question'], stats['mass']['avg_question']))
        print("{:35} {:<20.4f} {}".format(names[3], importance[3] * 100, stats['unique_words']))
        print("{:35} {:<20.4f} {}".format(names[4], importance[4] * 100, stats['unique_collocations']))
    else: #we know its only using first 4 features
        print("{:35} {:<20.4f} {:<50.4f} {:<20.4f}".format(names[0], importance[0] * 100,
                                                           stats['author']['avg_words_per_sent'],
                                                           stats['mass']['avg_words_per_sent']))
        print("{:35} {:<20.4f} {:<50.4f} {:<20.4f}".format(names[1], importance[1] * 100,
                                                           stats['author']['avg_sentence_length_variation'],
                                                           stats['mass']['avg_sentence_length_variation']))
        print("{:35} {:<20.4f} {:<50.4f} {:<20.4f}".format(names[2], importance[2] * 100,
                                                           stats['author']['avg_lex_diversity'] * 100,
                                                           stats['mass']['avg_lex_diversity'] * 100))
        print("{:35} {:<20.4f} {}".format(names[3], importance[3] * 100, stats['unique_words']))


#function used in baseline model to apply classifier to data
def PredictAuthorsBaseline(train, test, authors):

    # Create a Gaussian Classifier
    model = GaussianNB()

    # Train the model using the training sets
    train_authors = authors[0:len(train)]
    test_authors = authors[len(train):]
    model.fit(train, train_authors)

    # Predict Output
    predicted = model.predict(test)  # 0:Overcast, 2:Mild
    #print("Predicted Value: {}".format(predicted))

    tp=0
    tn =0
    fn=0
    fp=0
    for i in range(len(predicted)):
        if predicted[i]=="mass":
            if predicted[i] == test_authors[i]:
                tn+=1
            else:
                fn+=1
        else:
            if predicted[i]== test_authors[i]:
                tp+=1
            else:
                fp+=1

    print("tp: {}\tfn: {}\nfp: {}\ttn {}".format(tp, fn, fp, tn))
    print("accuracy: {}".format((tp+tn)/(tp+tn+fn+fp)))
    precision = (tp) / (tp + fp)
    recall = (tp) / (fn + tp)
    print("when it predicts yes, how often is it correct-- precision: {}".format(precision))
    print("when it's actually yes, how often does it predict yes -- recall: {}".format(recall))
    print("F-score: {}".format(2 * precision * recall / (precision + recall)))

# function to filter for ADJ/NN bigrams
def rightTypes(ngram):
    if '-pron-' in ngram or 't' in ngram:
        return False
    for word in ngram:
        if word in en_stopwords or word.isspace():
            return False
    acceptable_types = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    second_type = ('NN', 'NNS', 'NNP', 'NNPS')
    tags = nltk.pos_tag(ngram)
    if tags[0][1] in acceptable_types and tags[1][1] in second_type:
        return True
    else:
        return False

#this is the processing of features for the FINAL model
def all_features(data, id):
    results = {}


    posts = len(data)

    #split the data into author posts and the mass
    author_data = []
    mass_data = []
    for i, entry in enumerate(data):
        text = entry['text']
        author = entry['author']

        if author == id:
            author_data.append({"author": author, "text": text.strip()})
        else:
            mass_data.append({"author": author, "text": text.strip()})

    shuffle(mass_data)
    shuffle(author_data)

    train_data = mass_data[:math.floor(len(mass_data) *.75)] + author_data[:math.floor(len(author_data) *.75)]
    test_data = mass_data[math.floor(len(mass_data) * .75):] + author_data[math.floor(len(author_data) * .75):]

    #get set of words that are unique to the author being analyzed
    author_words = []
    mass_words = []
    author_bigrams = []
    mass_bigrams = []

    #use training data to find unique words and bigrams
    for i, entry in enumerate(train_data):
        text = entry['text']
        author = entry['author']

        words = word_tokenizer.tokenize(text.lower())

        if len(words) < 10:
            #pass on these entries that are too short, they are inconsistent and infrequent
            continue

        #get relevant bigrams using nltk collocations tool
        bigrams = nltk.collocations.BigramAssocMeasures()
        bigramFinder = nltk.collocations.BigramCollocationFinder.from_words(words)
        bigram_freq = bigramFinder.ngram_fd.items()
        bigramFreqTable = pd.DataFrame(list(bigram_freq), columns=['bigram', 'freq']).sort_values(by='freq', ascending=False)


        # filter bigrams further to get rid of ones with stop words or consist of odd forms
        filtered_bi = bigramFreqTable[bigramFreqTable.bigram.map(lambda x: rightTypes(x))]

        if author == id:
            for word in words:
                #filter out any words that contain underscores, etc
                if word.isalpha():
                    author_words.append(word)
            for index, row in filtered_bi.iterrows():
                author_bigrams.append(row['bigram'])
        else:
            for word in words:
                # filter out any words that contain underscores, etc
                if word.isalpha():
                    mass_words.append(word)
            for index, row in filtered_bi.iterrows():
                mass_bigrams.append(row['bigram'])

    #compute the intersections of sets to attain sets of unique words and collocations
    intersection = set(author_words).intersection(set(mass_words))
    unique_words = set(author_words) - intersection
    b_intersection = set(author_bigrams).intersection(set(mass_bigrams))
    unique_bigrams = set(author_bigrams) - b_intersection

    # find the terms with highest frequency of the unique words and bigrams
    unique_word_counts = [x for x in author_words if x in unique_words] #filter out non-unique words from author word list
    fdist_words = FreqDist(unique_word_counts)
    results['unique_words'] = fdist_words.most_common(10) #store 10 most common to return to user

    unique_bi_counts = [x for x in author_bigrams if x in unique_bigrams]#filter out non-unique bigrams from list
    fdist_bi = FreqDist(unique_bi_counts)
    results['unique_collocations'] = fdist_bi.most_common(10) #store 10 most commmon to return to user

    #recombine training and test data to iterate over full set
    data = train_data + test_data

    #build feature vectors for set
    fvs_lexical = np.zeros((len(data), 8), np.float64)
    authors = []
    author_data = []
    mass_data = []

    for i, entry in enumerate(data):
        text = entry['text']
        author = entry['author']
        #print(author, text)

        tokens = nltk.word_tokenize(text.lower())
        words = word_tokenizer.tokenize(text.lower())
        bigrams = list(nltk.bigrams(text))

        # filter out blog posts as garbage if too short, they cause execution errors and are anomalys
        if len(words) < 10:
            authors.append('mass')
            continue

        #lexical analysis
        sentences = sentence_tokenizer.tokenize(text)
        vocab = set(words)
        words_per_sentence = np.array([len(word_tokenizer.tokenize(s))
                                       for s in sentences])

        # average number of words per sentence
        fvs_lexical[i, 0] = words_per_sentence.mean()
        # sentence length variation
        fvs_lexical[i, 1] = words_per_sentence.std()
        # Lexical diversity
        fvs_lexical[i, 2] = len(vocab) / float(len(words))

        # unique words
        uniq_wc = len([word for word in words if word in unique_words and words.count(word) >1])
        uniq_wc = len([word for word in words if word in unique_words])
        fvs_lexical[i,3] = uniq_wc / len(words)

        uniq_bigram_ct = len([bigram for bigram in bigrams if bigram in unique_bigrams])
        fvs_lexical[i,4] = uniq_bigram_ct / len(bigrams)

        # punctuation analysis
        comma_ct = text.count(",")
        # semi_ct = text.count(";")
        exc_ct = text.count("!")
        q_ct = text.count("?")
        fvs_lexical[i,5] = comma_ct
        # fvs_lexical[i, 4] = semi_ct
        fvs_lexical[i, 6] = exc_ct
        fvs_lexical[i, 7] = q_ct

        #record the author of the passage for training
        if author == id:
            authors.append(author)
            author_data.append(list(fvs_lexical[i]))
        else:
            authors.append("mass")
            mass_data.append(list(fvs_lexical[i]))

    #compute writer and mass average statistics from test data
    author_stats = {}
    mass_stats = {}
    author_stats['avg_words_per_sent']=np.mean([x[0] for x in author_data])
    author_stats['avg_sentence_length_variation'] = np.mean([x[1] for x in author_data])
    author_stats['avg_lex_diversity'] = np.mean([x[2] for x in author_data])
    author_stats['avg_comma'] = np.mean([x[5] for x in author_data])
    author_stats['avg_exc'] = np.mean([x[6] for x in author_data])
    author_stats['avg_question'] = np.mean([x[7] for x in author_data])
    mass_stats['avg_words_per_sent'] = np.mean([x[0] for x in mass_data])
    mass_stats['avg_sentence_length_variation'] = np.mean([x[1] for x in mass_data])
    mass_stats['avg_lex_diversity'] = np.mean([x[2] for x in mass_data])
    mass_stats['avg_comma'] = np.mean([x[5] for x in mass_data])
    mass_stats['avg_exc'] = np.mean([x[6] for x in mass_data])
    mass_stats['avg_question'] = np.mean([x[7] for x in mass_data])
    results['author'] = author_stats
    results['mass'] = mass_stats

    fvs_lexical = whiten(fvs_lexical) #whiten the data to normalize columns

    #split into same training and test sets as before
    train = fvs_lexical[0:math.floor(posts * .75)]
    test = fvs_lexical[math.floor(posts * .75):]

    ans = PredictAuthors(train,test, authors,results)


# this function is the processing of features for the INTERMEDIATE model
def intermediate(data, id):
    results = {}
    all_words={}

    posts = len(data)
    shuffle(data)

    num_authors = len(all_words.keys())

    #get set of words that are unique to the author being analyzed
    author_words = []
    mass_words = []
    #only use first 75 % of posts for training, though
    for i, entry in enumerate(data[:math.floor(posts * .75)]):
        text = entry['text']
        author = entry['author']

        words = word_tokenizer.tokenize(text.lower())
        if author == id:
            for word in words:
                author_words.append(word)
        else:
            for word in words:
                mass_words.append(word)
    intersection = set(author_words).intersection(set(mass_words))
    unique_words = set(author_words) - intersection

    # find the terms with highest frequency of the unique words and bigrams
    unique_word_counts = [x for x in author_words if x in unique_words]  # filter out non-unique words from author word list
    fdist_words = FreqDist(unique_word_counts)
    results['unique_words'] = fdist_words.most_common(10)  # store 10 most common to return to user

    fvs_lexical = np.zeros((len(data), 4), np.float64)
    authors = []
    author_data = []
    mass_data = []

    for i, entry in enumerate(data):
        text = entry['text']
        author = entry['author']

        tokens = nltk.word_tokenize(text.lower())
        words = word_tokenizer.tokenize(text.lower())

        # filter out blog posts as garbage if too short, they cause execution errors and are anomalys
        if len(words) < 10:
            authors.append('mass')
            continue

        #lexical analysis
        sentences = sentence_tokenizer.tokenize(text)
        vocab = set(words)
        words_per_sentence = np.array([len(word_tokenizer.tokenize(s))
                                       for s in sentences])

        # average number of words per sentence
        fvs_lexical[i, 0] = words_per_sentence.mean()
        # sentence length variation
        fvs_lexical[i, 1] = words_per_sentence.std()
        # Lexical diversity
        fvs_lexical[i, 2] = len(vocab) / float(len(words))

        # unique words
        uniq_wc = len([word for word in words if word in unique_words and words.count(word) >1])
        uniq_wc = len([word for word in words if word in unique_words])
        fvs_lexical[i,3] = uniq_wc / len(words)


        #record the author of the passage for training
        if author == id:
            authors.append(author)
            author_data.append(list(fvs_lexical[i]))
        else:
            authors.append("mass")
            mass_data.append(list(fvs_lexical[i]))

    # compute writer and mass average statistics from test data
    author_stats = {}
    mass_stats = {}
    author_stats['avg_words_per_sent'] = np.mean([x[0] for x in author_data])
    author_stats['avg_sentence_length_variation'] = np.mean([x[1] for x in author_data])
    author_stats['avg_lex_diversity'] = np.mean([x[2] for x in author_data])
    mass_stats['avg_words_per_sent'] = np.mean([x[0] for x in mass_data])
    mass_stats['avg_sentence_length_variation'] = np.mean([x[1] for x in mass_data])
    mass_stats['avg_lex_diversity'] = np.mean([x[2] for x in mass_data])
    results['author'] = author_stats
    results['mass'] = mass_stats

    fvs_lexical = whiten(fvs_lexical)

    train = fvs_lexical[0:math.floor(posts * .75)]
    test = fvs_lexical[math.floor(posts * .75):]

    ans = PredictAuthors(train,test, authors,results)


#This function is the processing of features for the BASELINE model
def baseline(data, id):
    all_words = {}

    posts = len(data)
    shuffle(data)

    num_authors = len(all_words.keys())

    fvs_lexical = np.zeros((len(data), 3), np.float64)
    authors = []

    for i, entry in enumerate(data):
        text = entry['text']
        author = entry['author']

        tokens = nltk.word_tokenize(text.lower())
        words = word_tokenizer.tokenize(text.lower())

        # filter out blog posts as garbage if too short, they cause execution errors and are anomalys
        if len(words) < 10:
            authors.append('mass')
            continue

        # lexical analysis
        sentences = sentence_tokenizer.tokenize(text)
        vocab = set(words)
        words_per_sentence = np.array([len(word_tokenizer.tokenize(s))
                                       for s in sentences])

        # average number of words per sentence
        fvs_lexical[i, 0] = words_per_sentence.mean()
        # sentence length variation
        fvs_lexical[i, 1] = words_per_sentence.std()
        # Lexical diversity
        fvs_lexical[i, 2] = len(vocab) / float(len(words))

        # record the author of the passage for training
        if author == id:
            authors.append(author)
        else:
            authors.append("mass")

    fvs_lexical = whiten(fvs_lexical)

    train = fvs_lexical[0:math.floor(posts * .75)]
    test = fvs_lexical[math.floor(posts * .75):]

    results = PredictAuthorsBaseline(train, test, authors)


#this is MAIN EXECUTION
#load in data for n authors
if __name__ == "__main__":
    f= open("blogtext.csv", "r")
    f = [x.replace('\0', '') for x in f] # get rid of null chars causing errors

    data =[]
    n = 500#number of authors to use in the pool
    authors =0

    all_posts = {"authors": {}}

    reader = csv.reader(f)
    ct = 0
    print("loading data into dictionary")
    for row in reader:
        if ct ==0:
            #headers
            pass
        else:
            id = row[0]
            gender=row[1]
            age=row[2]
            topic=row[3]
            sign=row[4]
            date= row[5]
            text = row[6]
            if authors >= n and id not in all_posts["authors"].keys():
                continue

            if topic != "Student":
                continue

            if id in all_posts["authors"]:
                #author has already written something, so add
                pass
            else:
                #this is a new author to enter, so initialize entry
                all_posts["authors"][id] = {"info": {"gender": gender, "age": age, "sign": sign}, "posts":[]}
                authors += 1

            #add data into dictionalry and list
            all_posts["authors"][id]["posts"].append({"date": date, "topic": topic, "text": text.strip()})
            data.append({"author": id, "text": text.strip()})


        ct +=1
    print("done loading into dictionary")

    #pick the author with the most posts to be the one that we test upon
    max_posts=0
    max_id =0
    for author in all_posts["authors"].keys():
        num = len(all_posts['authors'][author]["posts"])
        if num > max_posts:
            max_posts = num
            max_id = author

    print("\n\nFINAL MODEL")
    all_features(data,max_id)
    print("\n\nINTERMEDIATE MODEL")
    intermediate(data, max_id)
    print("\n\nBASELINE MODEL")
    baseline(data,max_id)
