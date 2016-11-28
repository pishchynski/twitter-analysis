from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from stemming.porter2 import stem
import numpy as np
import csv
from gensim.models import word2vec
from gensim.models import doc2vec

def remove_usernames(str: "string with tweet message") -> "str without username":
    prepared_str = str[str.find("tweeted:") + 9 : ]
    if prepared_str.find("RT") != -1:
        prepared_str = prepared_str[prepared_str.find(":") + 2: ]
    return prepared_str


def remove_punctuation(msg: "string with tweet message"):
    soup = BeautifulSoup(msg, "html.parser")
    return soup.getText()


def remove_stop_words(str: "string with tweet message without punctuation"):
    words = str.split(" ")
    return [word for word in words if word not in stopwords.words('english')]


def stem_prepared_words(words: "list" "list with tweet words"):
    return list(map(lambda x: stem(x), words))


def run_prepare(input_filename: "path to tweets"):
    prep_tweets_list = []

    with open(input_filename, newline='', encoding='utf-8') as input_file:
        csv_reader = csv.reader(input_file, delimiter=',')
        for row in csv_reader:
            words_no_punct = remove_punctuation(row[-1])
            no_stop_words = remove_stop_words(words_no_punct)
            no_usernames = [word for word in no_stop_words if word.find('@') == -1]
            full_prepared_words = stem_prepared_words(no_stop_words)
            prep_tweets = row[0:-1]
            prep_tweets.append(full_prepared_words)
            prep_tweets_list.append(prep_tweets)
    return prep_tweets_list

def form_csv(input_filename):
    output_filename = input_filename + '.csv'
    prep_tweets = []
    tweet_id = 1
    with open(input_filename) as input_file:
        tweets = input_file.readlines()
        tweets_dict = {}
        prepared_str = """"""
        for msg in tweets:
            twilist = []
            twilist.append("""""")
            twilist.append(tweet_id)
            tweet_id += 1
            twilist.append('date')
            twilist.append('NO_QUERY')
            if msg[0] != '@':
                tweets_dict[prepared_str][-1] += msg
                continue
            username = msg[1:msg.find("""tweeted:""") - 1]
            twilist.append(username[1:])
            prepared_str = msg[msg.find("""tweeted:""") + 9:]
            if prepared_str.find("""RT""") != -1:
                prepared_str = prepared_str[prepared_str.find(""":""") + 2:]
            prepared_str = prepared_str.replace("\n", "")
            twilist.append(prepared_str)
            tweets_dict[prepared_str] = twilist
            print(tweet_id)
        prep_tweets = [[ v[0], v[1], v[2], v[3], v[4], k] for k, v in tweets_dict.items()]

    print("Prepared dataset size: ", len(prep_tweets))
    with open(output_filename, mode='w', newline='') as output_file:
        for tweet in prep_tweets:
            csv_writer = csv.writer(output_file, delimiter=""",""",  quoting=csv.QUOTE_ALL)
            csv_writer.writerow(tweet)


def build_word2vec_model(dataset, path):
    model = word2vec.Word2Vec(sentences=[tweet[5] for tweet in dataset], size=200, workers=2, min_count=1)
    # model.train([tweet[5] for tweet in test_dataset])
    # model.train([tweet[5] for tweet in main_dataset])
    model.save(path)
    return model

def build_doc2vec_model(train_dataset, test_dataset, main_dataset, path):
    model = doc2vec.Doc2Vec(documents=[tweet[5] for tweet in test_dataset], size=100, workers=2, min_count=1)
    # model.train([tweet[5] for tweet in test_dataset])
    # model.train([tweet[5] for tweet in main_dataset])
    model.save(path)
    return model


def load_word2vec_model(path):
    model = word2vec.Word2Vec.load(path)
    return model

def load_doc2vec_model(path):
    model = doc2vec.Doc2Vec.load(path)
    return model

def vectorize_dataset(dataset, model):
    return [model[word] for word in (line[5] for line in dataset)]

# run_prepare("twitter_#Trump.txt")
# run_prepare("twitter_Clinton.txt")
# run_prepare("twitter_Election.txt")
# run_prepare("twitter_Election2016.txt")
# run_prepare("twitter_ElectionResults.txt")
# run_prepare("twitter_Hillary.txt")
# run_prepare("twitter_Trump.txt")
# run_prepare("twitter_TrumpvsClinton.txt")
# run_prepare("twitter_USElection2016.txt")
# run_prepare("twitter_Выборы2016.txt")
# run_prepare("twitter_ВыборыСША.txt")
# run_prepare("twitter_Клинтон.txt")
# run_prepare("twitter_Трамп.txt")

# prepared_dataset = run_prepare("prepared_twitter_full.txt")
# form_csv("data/twitter_full.txt")

main_dataset = run_prepare('data/twitter_full.txt.csv')
train_dataset = run_prepare('training_data/training.1600000.processed.noemoticon.csv')
test_dataset = run_prepare('training_data/testdata.manual.2009.06.14.csv')

dataset = main_dataset + train_dataset + test_dataset

# build_word2vec_model(train_dataset=train_dataset, test_dataset=test_dataset, main_dataset=main_dataset, path="model/word2vec_tweets111")
model = build_word2vec_model(dataset=test_dataset, path="model/word2vec_tweets_full_200")

# model = load_word2vec_model(path="model/word2vec_tweets")
