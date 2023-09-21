import torch
from datasets import load_dataset
from tqdm import tqdm
import json

def classify_sst2():
    # Step 1: Import necessary built-in libraries
    from textblob import TextBlob

    # Step 2: Define a function that analyze the sentiment of a text
    def analyze_sentiment(text):
        sentiment = TextBlob(text).sentiment.polarity
        if sentiment > 0:
            return "positive"
        elif sentiment < 0:
            return "negative"
        else:
            return "neutral"

    dataset = load_dataset('sst2')
    val_dataset = dataset["validation"]
    acc = 0
    tot = 0
    for item in tqdm(val_dataset):
        tot += 1
        
        # Step 3: Define the text of movie review
        review = item['sentence']

        # Step 4: Print an answer in natural language using the knowledge and function defined above
        sentiment_result = analyze_sentiment(review)
        if sentiment_result == 'positive':
            label = 1
        else:
            label = 0
        if label == item['label']:
            acc += 1

    print(acc/tot)


def classify_cola():
    # Step 1: Import necessary built-in libraries
    import nltk

    # Step 2: Define a function that checks the grammar correctness of sentences
    def check_sentence(sentence):
        try:
            nltk.pos_tag(nltk.word_tokenize(sentence))
            return "acceptable"
        except:
            return "unacceptable"

    # Step 3: Define the input sentence
    dataset = load_dataset('glue', 'cola')
    val_dataset = dataset["validation"]
    acc = 0
    tot = 0
    for item in tqdm(val_dataset):
        tot += 1
        sentence = item['sentence']
        label = check_sentence(sentence)
        # Step 4: Classify the grammar of the sentence and print the result
        if item['label'] == 1:
            truth_label = 'acceptable'
        else:
            truth_label = 'unacceptable'
        if truth_label == label:
            acc += 1
       
    print(acc/tot)


def classifying_emotion():
    # Step 1: Import necessary built-in libraries
    from textblob import TextBlob

    # Step 2: Define a function that classify the emotion of the sentence
    def classify_emotion(sentence):
        blob = TextBlob(sentence)
        if blob.sentiment.polarity < 0:
            return "sad"
        elif blob.sentiment.polarity == 0:
            return "surprised"
        else:
            if "love" in sentence:
                return "love"
            elif "angry" in sentence:
                return "angry"
            elif "afraid" in sentence:
                return "afraid"
            else:
                return "happy"

    dataset = load_dataset('emotion')
    val_dataset = dataset["validation"]
    tot = 0
    acc = 0
    all_classes = ['sad', 'happy', 'love', 'angry', 'afraid', 'surprised']
    for item in tqdm(val_dataset):
        tot += 1
        # Step 3: Define a sentence 
        sentence = item['text']
       
        # Step 5: Print a classification answer in natural language using the defined function and sentence
        emotion = classify_emotion(sentence)
        if emotion == all_classes[item['label']]:
            acc += 1
    print(acc/tot)


def classify_review():
    # Step 1: Import necessary built-in libraries
    from textblob import TextBlob

    # Step 2: Define a function to analyze the stars of a given review
    def get_stars(review):
        analysis = TextBlob(review)
        # The polarity of TextBlob is in a range from -1 to 1, where -1 refers to negative sentiment and 1 to positive sentiment.
        # In order to map it to the star range of 1-5, we need to add 1 to make the polarity in 0-2 range, then divide it by 2 to make it in 0-1 range, 
        # and finally multiply it by (5 - 1) and add 1 to make it in 1-5 range.
        stars = round((analysis.sentiment.polarity + 1) / 2 * 4 + 1)
        return stars

    dataset = load_dataset('amazon_reviews_multi', 'en')
    val_dataset = dataset["validation"]
    tot = 0
    acc = 0
    classes = [1, 2, 3, 4, 5]  # Possible star rating

    for item in tqdm(val_dataset):
        tot += 1
        # Step 3: Specify a review to analyze
        review = item['review_title'] + item['review_body']

        # Step 4: Print the star from the analysis
        stars = get_stars(review)
        label = item['stars']
        if stars == label:
            acc += 1
    print(acc/tot)



def classify_hsd():
    # Step 1: Import necessary built-in libraries
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from nltk.tokenize import word_tokenize
    import nltk
    nltk.download('vader_lexicon')

    # Step 2: Define a function to check feelings of sentences
    def hate_speech_check(speech):
        words = word_tokenize(speech)
        # We use nltk library to determine the feelings of the speech
        sid = SentimentIntensityAnalyzer()
        sentiment = sid.polarity_scores(speech)
        # Considered hate speech if compound sentiment is less than -0.3
        if sentiment['compound'] < -0.3:
            return True
        else:
            return False

    with open('../hate-speech-dataset/test_sample.json', 'r') as f:
        data = json.load(f)
    tot = 0
    acc = 0
    for item in data:
        tot += 1
        # Step 3: Define dictionaries storing the speech
        speech = item['sentence']

        # Step 4: Print an result using the defined function and speech
        is_hate_speech = hate_speech_check(speech)
        if is_hate_speech:
            if item['label'] == 'hate':
                acc += 1
        else:
            if item['label'] != 'hate':
                acc += 1
    print(acc/tot)



def classify_sbic():
    # Step 1: Import necessary built-in libraries
    import re

    # Step 2: Define a dictionary storing keywords which identify offensiveness
    keywords = {
        "Yes": ["hate", "kill", "racist", "sexist", "explicit"], # A list containing some offensive words for example
        "Maybe": ["idiot", "stupid", "dumb", "weirdo"], # A list containing some maybe offensive words for example
        "No": ["happy", "love", "beautiful", "amazing", "fantastic"] # A list containing some positive words for example
    }

    # Step 3: Define a function that classify a post
    def classify_post(post, keywords):
        word_occurrences = {"Yes": 0, "Maybe": 0, "No": 0} # Initialize a dictionary to store the number of occurrences 
        for keyword in re.findall(r'\w+', post): # Split the post into words
            for key in keywords.keys():
                if keyword.lower() in keywords[key]:
                    word_occurrences[key] += 1
            
        max_key = max(word_occurrences, key=word_occurrences.get) # Find the key with maximum occurrences
        if word_occurrences[max_key] > 0:
            return max_key
        else:
            return "No"    
    
    dataset = load_dataset('social_bias_frames')
    val_dataset = dataset["validation"]
    tot = 0
    acc = 0
    for item in tqdm(val_dataset):
        tot += 1
        # Step 4: Define dictionaries storing the post
        post = item['post']
        truth = item['offensiveYN']
        
        if truth == '1.0':
            truth = 'Yes'
        elif truth == '0.5':
            truth = 'Maybe'
        else:
            truth = 'No'

        # Step 5: Print an answer in natural language using the knowledge and function defined above
        classification = classify_post(post, keywords)
        if truth == classification:
            acc += 1
    print(acc/tot)


def classify_agnews():
    # Step 1: Import necessary built-in libraries
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB

    # Step 2: Define the training dataset for the classification model
    news_data = {
        "world news": [
            "The 193-member United Nations General Assembly elected the United Arab Emirates, Brazil, Albania, Gabon, and Ghana to serve on the U.N. Security Council for two years.",
            "North Korea announced that it has test launched series of tactical guided missiles.",
            "European Union leaders reach agreement on a new tax proposal that affects multinational companies."
        ],
        "sport news": [
            "The German team edged out Portugal in a high-scoring Euro 2020 group-stage game.",
            "Serena Williams withdraws from Wimbledon due to injury.",
            "Tokyo Olympic organizers announced a limit on spectators in the upcoming summer games."
        ],
        "business news": [
            "Airline companies see a rise in domestic travel bookings as Covid-19 restrictions begin to ease.",
            "Shares of AMC Entertainment soared due to coordinated trading by individual investors.",
            "Microsoft Corp is set to acquire LinkedIn Corp for $26.2 billion in cash."
        ],
        "technology news": [
            "Apple Inc. announces new privacy features for its upcoming iOS 15 operating system.",
            "Facebook Inc tests new tool for users to manage content from their News Feed.",
            "Google's proposed acquisition of Fitbit to be closely scrutinized by the EU."
        ]
    }

    # Step 3: Prepare the data
    X_train = []
    y_train = []
    for category, news_list in news_data.items():
        for news in news_list:
            X_train.append(news)
            y_train.append(category)

    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train)

    # Step 4: Train the classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    classes = ['world news', 'sport news', 'business news', 'technology news']
    dataset = load_dataset('ag_news', cache_dir = '/shared/jiaxin/.cache')
    val_dataset = dataset["test"]
    tot = 0
    acc = 0
    for item in tqdm(val_dataset):
        tot += 1
        # Step 5: Define the news piece to be classified
        piece_of_news = item['text']
        # Step 6: Make the prediction
        X_test = vectorizer.transform([piece_of_news])
        predicted_category = clf.predict(X_test)[0]

        if predicted_category == classes[item['label']]:
            acc += 1
    print(acc/tot)
