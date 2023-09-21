
import torch
from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification
from transformers import DebertaTokenizer, DebertaModel, DebertaForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
from tqdm import tqdm
import json
# Load pre-trained RoBERTa model and tokenizer
device = 'cuda'
# Load Roberta:
model = RobertaForSequenceClassification.from_pretrained("luohy/ESP-roberta-large")
tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

# Load Deberta:
model = DebertaForSequenceClassification.from_pretrained("luohy/ESP-deberta-large")
tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-large")

model.eval()  # Set to evaluation mode
model = model.to(device)
softmax = torch.nn.Softmax(dim=0)

# General Functions that are used for entailment model:
def get_entailment_score(hypothesis, model, tokenizer):
    with torch.no_grad():
        proposition = f'{hypothesis} is entailed by .'
        inputs = tokenizer(proposition,  return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        outputs = model(**inputs)['logits'][0]
        ent = outputs[0]
        contra = outputs[2]
        return ent, contra
    
def entailment(hypothesis, premise, model, tokenizer, reduce=(0,0), return_val=False):
    with torch.no_grad():
        proposition = f'{hypothesis} is entailed by {premise}.'
        inputs = tokenizer(proposition,  return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        outputs = model(**inputs)['logits'][0]
        ent_label = int((outputs[0]-reduce[0]) > (outputs[2]-reduce[1]))
        if ent_label == 1:
            if return_val:
                val = outputs[0] - reduce[0]
                return 'yes', val
            return 'yes'
        else:
            if return_val:
                val = outputs[0] - reduce[0]
                return 'no', val
            return 'no'
            
def tree_predict(sentence, criterions, tree, model, tokenizer, score_list, return_val=False):
    node = tree['root']
    score_tuple = score_list[node]
    tot_num = 0
    tot_val = 0.0
    while node not in POSSIBLE_CLASSES:
        score_tuple = score_list[node]
        if not return_val:
            ent_label = entailment(criterions[node], sentence, model, tokenizer, score_tuple, return_val)
        else:  
            ent_label, val = entailment(criterions[node], sentence, model, tokenizer, score_tuple, return_val)
            tot_num += 1
            tot_val += val
        node = tree[node][ent_label]
    if return_val:
        return node, tot_val/tot_num
    return node

# SST2 decision tree and possible classes
POSSIBLE_CLASSES = ["positive", 'negative']
def get_decision_tree_sst2(which_tree='gpt4'):
    if which_tree == 'human':
        # Human crafted tree
        # Step 1: define criterions of the decision tree.
        criterions = {
            'is_interesting':'This movie is interesting',
            'is_good_script':'The movie has a good script',
            'is_good_character':'The characters are awsome',
            'is_wise': 'This movie is wise'
        }

        # Step 2: define the balanced decision tree for this classification task
        tree = {
            'root': 'is_interesting',
            'is_interesting': {'yes': 'is_good_script', 'no': 'is_wise'},
            'is_good_script': {'yes': 'positive', 'no': 'is_good_character'},
            'is_good_character': {'yes': 'positive', 'no': 'negative'},
            'is_wise': {'yes': 'positive', 'no': 'negative'}
        }
    else:
        # Automatic Tree
        # Step 1: define criterions of the decision tree
        criterions = {
            'positive_adjectives': 'The review uses positive adjectives',
            'negative_adjectives': 'The review uses negative adjectives',
            'positive_director_mention': 'The review mentions the director\'s name positively',
            'negative_cast_comments': 'The review comments negatively on the cast'
        }

        # Step 2: define the balanced decision tree for this classification task
        tree = {
            'root': 'positive_adjectives',
            'positive_adjectives': {'yes': 'positive', 'no': 'negative_adjectives'},
            'negative_adjectives': {'yes': 'negative', 'no': 'positive_director_mention'},
            'positive_director_mention': {'yes': 'positive', 'no': 'negative_cast_comments'},
            'negative_cast_comments': {'yes': 'negative', 'no': 'positive'}
        }
    all_scores = {}
    for key in criterions.keys():
        sentence = criterions[key]
        score = get_entailment_score(sentence, model, tokenizer)
        all_scores[key] = score
    return criterions, tree, all_scores

    

def get_sst2_tree_score():
    dataset = load_dataset('sst2')
    val_dataset = dataset["validation"]
    criterions, tree, all_score = get_decision_tree_sst2()
    tot = 0
    acc = 0
    for item in tqdm(val_dataset):
        tot += 1
        sentence = item['sentence']
        
        label, val = tree_predict(sentence, criterions, tree, model, tokenizer, all_score, True)
        print(sentence)
        print(POSSIBLE_CLASSES[item['label']])
        print(label)
        if label == POSSIBLE_CLASSES[1-item['label']]:
            acc += 1
    print(acc/tot)


# COLA decision tree and testing
POSSIBLE_CLASSES = ["acceptable", 'unacceptable']
def get_decision_tree_cola(which_tree='gpt4'):
    if which_tree == 'gpt4':
        # GPT 4
        # Step 1: define criterions of the decision tree.
        criterions = {
            0:'This sentence is grammatically correct',
            1:'The sentence does not contain any spelling errors',
            2:'The sentence uses punctuation correctly',
            3:'This sentence is semantically clear'
        }

        # Step 2: define the Decision Tree for classification
        tree = {
            'root': 0,
            0: {'yes': 1, 'no': 3},
            1: {'yes': 'acceptable', 'no': 2},
            2: {'yes': 'acceptable', 'no': 'unacceptable'},
            3: {'yes': 2, 'no': 'unacceptable'}
        }
    elif which_tree == 'gpt3.5'
        # GPT3.5
        # Step 1: define criterions of the decision tree.
        criterions = {
            'is_grammatically_correct': 'The sentence is grammatically correct',
            'is_clear': 'The sentence is clear',
            'has_minor_errors': 'The sentence has minor errors',
            'has_major_errors': 'The sentence has major errors'
        }

        # Step 2: define the balanced decision tree for this classification task
        tree = {
            'root': 'is_grammatically_correct',
            'is_grammatically_correct': {'yes': 'is_clear', 'no': 'has_minor_errors'},
            'is_clear': {'yes': 'acceptable', 'no': 'has_major_errors'},
            'has_minor_errors': {'yes': 'acceptable', 'no': 'unacceptable'},
            'has_major_errors': {'yes': 'unacceptable', 'no': 'acceptable'}
        }
    elif which_tree == 'human':
        # Step 1: define criterions of the decision tree.
        criterions = {
            'has_subject': 'The sentence has subject',
            'has_verb': 'The sentence has verb',
            'punctuation_correct':'The sentence has proper punctuations',
            'pronouns_reference_match':'This sentence has matched pronouns and references',
            'subject_verb_match':'The sentence has matched subject and verb',
        }

        # Step 2: define the balanced decision tree for this classification task
        tree = {
            'root': 'has_subject',
            'has_subject': {'yes': 'has_verb', 'no': 'unacceptable'},
            'has_verb': {'yes': 'punctuation_correct', 'no': 'unacceptable'},
            'punctuation_correct': {'yes': 'pronouns_reference_match', 'no': 'unacceptable'},
            'pronouns_reference_match': {'yes': 'subject_verb_match', 'no': 'unacceptable'},
            'subject_verb_match':{'yes': 'acceptable', 'no': 'unacceptable'}
        }
    all_scores = {}
    for key in criterions.keys():
        sentence = criterions[key]
        score = get_entailment_score(sentence, model, tokenizer)
        all_scores[key] = score
    return criterions, tree, all_scores

# baseline of cola using entailment model
def get_decision_cola_baseline():
    # Step 1: define criterions of the decision tree.
    criterions = {
        0:'This sentence is acceptable',
    }

    # Step 2: define the Decision Tree for classification
    tree = {
        'root': 0,
        0: {'yes': 'acceptable', 'no':'unacceptable'},
    }

    all_scores = {}
    for key in criterions.keys():
        sentence = criterions[key]
        score = get_entailment_score(sentence, model, tokenizer)
        all_scores[key] = score
    return criterions, tree, all_scores

def get_cola_tree_score():
    dataset = load_dataset('glue', 'cola')
    val_dataset = dataset["validation"]
    criterions, tree, all_score = get_decision_tree_cola()
    tot = 0
    acc = 0
    for item in tqdm(val_dataset):
        tot += 1
        sentence = item['sentence']
        
        label, val = tree_predict(sentence, criterions, tree, model, tokenizer, all_score, True)
        if label == POSSIBLE_CLASSES[item['label']]:
            acc += 1
    print(acc/tot)



# Emotion Classification tree and testing
POSSIBLE_CLASSES = ['I feel sad', 'I feel happy', 'I feel love', 'I feel angry', 'I feel afraid', 'I feel surprised']
def get_emotion_tree(which_tree='gpt4'):
    if which_tree == 'gpt4':
        # GPT4
        # Step 1: define criterions of the decision tree.
            criterions = {
                'is_positive':'This feeling is positive',
                'is_sad':'I feel sad',
                'is_angry':'I feel angry',
                'is_afraid':'I feel afraid',
                'is_happy':'I feel happy',
                'is_love':'I feel love',
                'is_surprised':'I feel surprised'
            }

            # Step 2: define the balanced decision tree for this classification task
            tree = {
                'root': 'is_positive',
                'is_positive': {'yes': 'is_happy', 'no': 'is_sad'},
                'is_happy': {'yes': 'I feel happy', 'no': 'is_love'},
                'is_love': {'yes': 'I feel love', 'no': 'is_surprised'},
                'is_surprised': {'yes': 'I feel surprised', 'no': 'I feel happy'},
                'is_sad': {'yes': 'I feel sad', 'no': 'is_angry'},
                'is_angry': {'yes': 'I feel angry', 'no': 'is_afraid'},
                'is_afraid': {'yes': 'I feel afraid', 'no': 'I feel sad'}
            }
    elif which_tree == 'gpt3.5':
        # GPT3.5
        # Step 1: define criterions of the decision tree.
        criterions = {
            'is_positive':'The sentence expresses positive emotion',
            'is_love':'The sentence expresses love',
            'is_happy':'The sentence expresses happiness',
            'is_negative':'The sentence expresses negative emotion',
            'is_anger':'The sentence expresses anger',
            'is_sad':'The sentence expresses sadness'
        }

        # Step 2: define the balanced decision tree for this classification task
        tree = {
            'root': 'is_positive',
            'is_positive': {'yes': 'is_love', 'no': 'is_negative'},
            'is_love': {'yes': 'I feel love', 'no': 'is_happy'},
            'is_happy': {'yes': 'I feel happy', 'no': 'I feel surprised'},
            'is_negative': {'yes': 'is_anger', 'no': 'is_sad'},
            'is_anger': {'yes': 'I feel angry', 'no': 'is_sad'},
            'is_sad': {'yes': 'I feel sad', 'no': 'I feel afraid'}
        }
    elif which_tree == 'human':
        # Human Prompt
        # Step 1: define criterions of the decision tree.
        criterions = {
            'positive_words': 'The sentence includes positive words',
            'intensifiers': 'The sentence includes intensifiers, adverb or exclamation point',
            'unexpected': 'The sentence includes words that are synonyms or antonyms of unexpected',
            'fear': 'The sentence includes words that are synonyms or antonyms of fear',
            'upset': 'The sentence includes words that are synonyms or antonyms of upset'
        }

        # Step 2: define the balanced decision tree for this classification task
        tree = {
            'root': 'positive_words',
            'positive_words': {'yes': 'intensifiers', 'no': 'fear'},
            'intensifiers': {'yes': 'unexpected', 'no': 'I feel happy'},
            'unexpected': {'yes': 'I feel surprised', 'no': 'I feel love'},
            'fear': {'yes': 'I feel afraid', 'no': 'upset'},
            'upset': {'yes': 'I feel sad', 'no': 'I feel angry'}
        }
    all_scores = {}
    for key in criterions.keys():
        sentence = criterions[key]
        score = get_entailment_score(sentence, model, tokenizer)
        all_scores[key] = score
    return criterions, tree, all_scores

# Another side-note: generate a tree for each emotion and get the score for each class; choose the highest class label as the result
def get_happy_tree():
    # Step 1: define criterions of the decision tree.
    criterions = [
        'I am feeling good',
        'I have a positive attitude',
        'I am smiling',
        'I am excited'
    ]

    # Step 2: define the balanced decision tree for this classification task
    tree = {
        'root': 0,
        0: {'yes': 1, 'no': 2},
        1: {'yes': 'Yes', 'no': 3},
        2: {'yes': 3, 'no': 'No'},
        3: {'yes': 'Yes', 'no': 'No'}
    }

    all_scores = []
    for sentence in criterions:
        score = get_entailment_score(sentence, model, tokenizer)
        all_scores.append(score)
        del score
    return criterions, tree, all_scores
def get_sad_tree():
    # Step 1: define criterions of the decision tree.
    criterions = [
        'I am feeling depressed',
        'I am not feeling good',
        'I am missing someone',
        'I feel like crying'
    ]

    # Step 2: define the balanced decision tree for this classification task
    tree = {
        'root': 0,
        0: {'yes': 1, 'no': 3},
        1: {'yes': 'Yes', 'no': 2},
        2: {'yes': 'Yes', 'no': 'No'},
        3: {'yes': 'Yes', 'no': 'No'}
    }
    

    all_scores = []
    for sentence in criterions:
        score = get_entailment_score(sentence, model, tokenizer)
        all_scores.append(score)
        del score
    return criterions, tree, all_scores

def get_love_tree():
    # Step 1: define criterions of the decision tree.
    criterions = [
        'I mention love in this sentence',
        'I talk about affection in this sentence',
        'I express happiness in this sentence',
        'I mention someone special in this sentence'
    ]

    # Step 2: define the balanced decision tree for this classification task
    tree = {
        'root': 0,
        0: {'yes': 1, 'no': 2},
        1: {'yes': 'Yes', 'no': 3},
        2: {'yes': 'Yes', 'no': 'No'},
        3: {'yes': 'Yes', 'no': 'No'}
    }

    all_scores = []
    for sentence in criterions:
        score = get_entailment_score(sentence, model, tokenizer)
        all_scores.append(score)
        del score
    return criterions, tree, all_scores

def get_anger_tree():
    # Step 1: define criterions of the decision tree.
    criterions = [
        'I am irritated',
        'I am frustrated',
        'I am outraged',
        'I am moody'
    ]

    # Step 2: define the balanced decision tree for this classification task
    tree = {
        'root': 0,
        0: {'yes': 1, 'no': 2},
        1: {'yes': 'Yes', 'no': 3},
        2: {'yes': 'Yes', 'no': 'No'},
        3: {'yes': 'Yes', 'no': 'No'}
    }

    all_scores = []
    for sentence in criterions:
        score = get_entailment_score(sentence, model, tokenizer)
        all_scores.append(score)
        del score
    return criterions, tree, all_scores

def get_fear_tree():
    # Step 1: define criterions of the decision tree.
    criterions = [
        'I feel scared',
        'I am frightened',
        'I feel terrified',
        'There is danger'
    ]
    
    # Step 2: define the balanced decision tree for this classification task
    tree = {
        'root': 0,
        0: {'yes': 1, 'no': 2},
        1: {'yes': 'Yes', 'no': 'No'},
        2: {'yes': 3, 'no': 'No'},
        3: {'yes': 'Yes', 'no': 'No'}
    }

    all_scores = []
    for sentence in criterions:
        score = get_entailment_score(sentence, model, tokenizer)
        all_scores.append(score)
        del score
    return criterions, tree, all_scores

def get_surprise_tree():
    # Step 1: define criterions of the decision tree.
    criterions = [
        'I did not expect this',
        'This is unusual',
        'I am startled',
        'This is a surprise'
    ]

    # Step 2: define the balanced decision tree for this classification task
    tree = {
        'root': 0,
        0: {'yes': 1, 'no': 2},
        1: {'yes': 'Yes', 'no': 3},
        2: {'yes': 'Yes', 'no': 'No'},
        3: {'yes': 'Yes', 'no': 'No'}
    }
    
    all_scores = []
    for sentence in criterions:
        score = get_entailment_score(sentence, model, tokenizer)
        all_scores.append(score)
    return criterions, tree, all_scores

# test emotion by using only one tree
def get_full_tree_emotion():
    dataset = load_dataset('emotion')
    val_dataset = dataset["validation"]
    criterions, tree, all_score = get_emotion_tree()
    tot = 0
    acc = 0
    for item in tqdm(val_dataset):
        tot += 1
        sentence = item['text']
        
        label, val = tree_predict(sentence, criterions, tree, model, tokenizer, all_score, True)
        if label == POSSIBLE_CLASSES[item['label']]:
            acc += 1
    print(acc/tot)

# Use multiple trees for each class and choose the highest-scoring class
def get_all_tree_emotion():
    EMOTION_CLASSES = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    criterions_sad, tree_sad, score_sad = get_sad_tree()
    criterions_happy, tree_happy, score_happy = get_happy_tree()
    criterions_love, tree_love, score_love = get_love_tree()
    criterions_anger, tree_anger, score_anger = get_anger_tree()
    criterions_fear, tree_fear, score_fear = get_fear_tree()
    criterions_surprise, tree_surprise, score_surprise = get_surprise_tree()
    all_criterion = [criterions_sad, criterions_happy,criterions_love, criterions_anger,criterions_fear,criterions_surprise]
    all_tree = [tree_sad, tree_happy, tree_love, tree_anger,tree_fear, tree_surprise]
    all_score = [score_sad, score_happy, score_love, score_anger, score_fear, score_surprise]
    tot = 0
    acc = 0
    for item in tqdm(val_dataset):
        tot += 1
        sentence = item['text']
        correct_emotions = []
        emotion_val = []
        for criterions, tree, score, emotion in zip(all_criterion, all_tree,all_score, EMOTION_CLASSES):
            label, val = tree_predict(sentence, criterions, tree, model, tokenizer, score, True)
            if label == 'Yes':
                correct_emotions.append(emotion)
            emotion_val.append(val)
   
        if len(correct_emotions) == 1:
            label = correct_emotions[0]
        elif len(correct_emotions) > 1:
            init = -100
            for num, emotion in enumerate(EMOTION_CLASSES):
                if emotion in correct_emotions and emotion_val[num] > init:
                    label = emotion
                    init = emotion_val[num]
        else:
            index = torch.argmax(torch.tensor(emotion_val))
            label = EMOTION_CLASSES[index]
       
        if label == EMOTION_CLASSES[item['label']]:
            acc += 1
    print(acc/tot)
    with open('./results/emotion_tree', 'w') as f:
        f.write(str(acc/tot))

# Tree and testing for amazon review
POSSIBLE_CLASSES = ['1', '2', '3', '4', '5']
def get_amazon_review_tree(which_tree='gpt4'):
    if which_tree == 'gpt4':
        # Step 1: define criterions of the decision tree.

        criterions = {
            'is_very_satisified':'This review is very satisfied',
            'is_superior_quality':'The product is of superior quality',
            'is_moderately_satisfied':'This review is moderately satisfied',
            'is_critic':'The review contains criticism',
            'is_satisfied':'The review is satisfied',
            'is_improvable':'The product has features to improve',
            'is_unsatisfied':'The review is unsatisfied',
            'is_extremely_negative':'The review contains extremely negative words'
        }

        # Step 2: define the balanced decision tree for this classification task
        tree = {
            'root': 'is_very_satisified',
            'is_very_satisified': {'yes': 'is_superior_quality', 'no': 'is_moderately_satisfied'},
            'is_superior_quality': {'yes': '5', 'no': '4'},
            'is_moderately_satisfied': {'yes': 'is_critic', 'no': 'is_satisfied'},
            'is_critic': {'yes': '3', 'no': '4'},
            'is_satisfied': {'yes': 'is_improvable', 'no': '2'},
            'is_improvable': {'yes': '3', 'no': '2'},
            'is_unsatisfied': {'yes': 'is_extremely_negative', 'no': '2'},
            'is_extremely_negative': {'yes': '1', 'no': '2'}
        }
    elif which_tree == 'human':
        # Step 1: define criterions of the decision tree.
        criterions = {
            'extreme_word': 'The review includes intensifiers or exclamation point',
            'positive': 'The review includes positive words or sentences',
            'both_positive_negative':'The review includes both positive and negative sentences',
        }

        # Step 2: define the balanced decision tree for this classification task
        tree = {
            'root': 'extreme_word',
            'extreme_word': {'yes': 'positive', 'no': 'both_positive_negative'},
            'positive': {'yes': '5', 'no': '1'},
            'both_positive_negative': {'yes': '3', 'no': 'positive'},
            'positive': {'yes': '4', 'no': '2'}
        }
    elif which_tree == 'gpt3.5':
        # GPT3.5
        # Step 1: define criterions of the decision tree.
        criterions = {
            'is_negative_sentiment':'This review has negative sentiment',
            'is_short_review':'This review is short',
            'has_positive_keywords':'This review has positive keywords',
            'has_negative_keywords':'This review has negative keywords'
        }

        # Step 2: define the balanced decision tree for this classification task
        tree = {
            'root': 'is_negative_sentiment',
            'is_negative_sentiment': {'yes': '1', 'no': 'is_short_review'},
            'is_short_review': {'yes': '3', 'no': 'has_positive_keywords'},
            'has_positive_keywords': {'yes': '5', 'no': 'has_negative_keywords'},
            'has_negative_keywords': {'yes': '2', 'no': '4'}
        }
    all_scores = {}
    for key in criterions.keys():
        sentence = criterions[key]
        score = get_entailment_score(sentence, model, tokenizer)
        all_scores[key] = score
    return criterions, tree, all_scores

def get_full_tree_amazon():

    dataset = load_dataset('amazon_reviews_multi', 'en')
    val_dataset = dataset["validation"]
    criterions, tree, all_score = get_amazon_review_tree()
    tot = 0
    acc = 0
    for item in tqdm(val_dataset):
        tot += 1
        sentence = item['review_title'] + item['review_body']
        label, val = tree_predict(sentence, criterions, tree, model, tokenizer, all_score, True)
        if label == POSSIBLE_CLASSES[item['stars']-1]:
            acc += 1
    print(acc/tot)


POSSIBLE_CLASSES = ['this is hate', 'this is noHate']
def get_hate_tree(which_tree='gpt4'):
    if which_tree=='gpt4':
        # GPT4
        # # Step 1: define criterions of the decision tree.
        criterions = {
            'is_aggressive':'This speech contains aggressive language',
            'is_targeting_group':'This speech targets a specific group',
            'is_explicit_discrimination':'This speech contains explicit discrimination',
            'has_covert_prejudice': 'This speech has covert prejudice or bias'
        }

        # Step 2: define the balanced decision tree for this classification task
        tree = {
            'root': 'is_aggressive',
            'is_aggressive': {'yes': 'is_targeting_group', 'no': 'has_covert_prejudice'},
            'is_targeting_group': {'yes': 'this is hate', 'no': 'is_explicit_discrimination'},
            'is_explicit_discrimination': {'yes': 'this is hate', 'no': 'this is noHate'},
            'has_covert_prejudice': {'yes': 'this is hate', 'no': 'this is noHate'}
        }
    elif which_tree == 'human':
        # Human Tree
        # Step 1: define criterions of the decision tree.
        criterions = {
            'oi_words': 'The speech includes offensive words about minorities',
            'identity_words': 'The sentence includes mention of identities',
            'swear_words': 'The sentence includes swear words',
            'negative_words':'The sentence includes negative words',
        }

        # Step 2: define the balanced decision tree for this classification task
        tree = {
            'root': 'oi_words',
            'oi_words': {'yes': 'this is hate', 'no': 'identity_words'},
            'identity_words': {'yes': 'swear_words', 'no': 'this is noHate'},
            'swear_words': {'yes': 'this is hate', 'no': 'negative_words'},
            'negative_words': {'yes': 'this is hate', 'no': 'this is noHate'},
        }
    elif which_tree=='gpt3.5':
        # GPT3.5
        # Step 1: define criterions of the decision tree.
        criterions = {
            'is_explicit_hate':'This speech contains explicit hate speech',
            'is_derogatory_language':'This speech contains derogatory language'
        }

        # Step 2: define the balanced decision tree for this classification task
        tree = {
            'root': 'is_explicit_hate',
            'is_explicit_hate': {'yes': 'this is hate', 'no': 'is_derogatory_language'},
            'is_derogatory_language': {'yes': 'this is hate', 'no': 'this is noHate'}
        }
    all_scores = {}
    for key in criterions.keys():
        sentence = criterions[key]
        score = get_entailment_score(sentence, model, tokenizer)
        all_scores[key] = score
    return criterions, tree, all_scores


def get_hate_dataset():
    import os
    import csv
    import json
    # Paths to CSV file and test set folder
    csv_file_path = '../hate-speech-dataset/annotations_metadata.csv'
    test_folder_path = '../hate-speech-dataset/sampled_test/'

    # Read the CSV file and create a dictionary of file IDs and labels
    file_labels = {}
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            print(row)
            file_id = row['file_id']
            label = row['label']
            file_labels[file_id] = label

    # Iterate over files in the test set folder
    data = []
    for filename in os.listdir(test_folder_path):
        if filename.endswith('.txt'):
            file_id = filename.split('.')[0]
            file_path = os.path.join(test_folder_path, filename)
            
            # Check if the file ID has a corresponding label
            if file_id in file_labels:
                label = file_labels[file_id]
                with open(file_path, 'r') as txt_file:
                    sentence = txt_file.read().strip()
                    print(f'File: {filename}\nLabel: {label}\nSentence: {sentence}\n')
                    new_dict = {}
                    new_dict['file'] = filename
                    new_dict['label'] = label
                    new_dict['sentence'] = sentence
                    data.append(new_dict)
            else:
                print(f'No label found for File: {filename}\n')
    with open('../hate-speech-dataset/test_sample.json', 'w') as f:
        json.dump(data, f)

def get_full_tree_hate_speech():
    with open('../hate-speech-dataset/test_sample.json', 'r') as f:
        data = json.load(f)
    criterions, tree, all_score = get_hate_tree()
    tot = 0
    acc = 0
    for item in tqdm(data):
        tot += 1
        sentence = item['sentence']
        
        label, val = tree_predict(sentence, criterions, tree, model, tokenizer, all_score, True)
        
        if label == 'this is ' + item['label']:
            acc += 1
    print(acc/tot)
# get_full_tree_hate_speech()


def get_baseline_hate_speech():
    with open('../hate-speech-dataset/test_sample.json', 'r') as f:
        data = json.load(f)
    tot = 0
    acc = 0
    hate_tuple =  get_entailment_score('this is hate ', model, tokenizer)
    nonhate_tuple = get_entailment_score('this is not hate ', model, tokenizer)
    for item in tqdm(data):
        tot += 1
        sentence = item['sentence']
        
        hatelabel, hateval = entailment('this is hate ', sentence, model, tokenizer, hate_tuple, True)
        nohatelabel, nohateval = entailment('this is not hate ', sentence, model, tokenizer, nonhate_tuple, True)
        if hateval > nohateval:
            label = 'hate'
        else:
            label = 'noHate'
        
        if label == item['label']:
            acc += 1
    print(acc/tot)



POSSIBLE_CLASSES = ['Yes','Maybe', 'No']
def get_bias_offense_tree(which_tree='gpt4', use_model='deberta'):
    if which_tree=='gpt4':
        if use_model=='roberta':
            # Step 1: define criterions of the decision tree.
            criterions = {
                'is_hate_speech':'This post contains hate speech',
                'is_disrespectful':'This post is disrespectful',
                'has_strong_language':'This post contains strong language',
            }

            # Step 2: define the balanced decision tree for this classification task
            tree = {
                'root': 'is_hate_speech',
                'is_hate_speech': {'yes': 'Yes', 'no': 'is_disrespectful'},
                'is_disrespectful': {'yes': 'Maybe', 'no': 'has_strong_language'},
                'has_strong_language': {'yes': 'Maybe', 'no': 'No'},
            }
        else:
            # Step 1: define criterions of the decision tree.
            criterions = {
                'is_politically_incorrect':'This is politically incorrect',
                'includes_offensive_words':'This includes offensive words',
                'is_discriminating':'This is discriminative'
            }

            # Step 2: define the balanced decision tree for this task.
            tree = {
                'root': 'is_politically_incorrect',
                'is_politically_incorrect': {'yes': 'includes_offensive_words', 'no': 'is_discriminating'},
                'includes_offensive_words': {'yes': 'Yes', 'no': 'Maybe'},
                'is_discriminating': {'yes': 'Maybe', 'no': 'No'}
            }
    elif which_tree == 'human'
        # Human Tree
        # Step 1: define criterions of the decision tree.
        criterions = {
            'racist_words': 'The post includes racist words',
            'sexist_words': 'The post includes sexist words',
            'swear_words': 'The post includes swear words',
            'identity_words': 'The post includes mention of identities',
            'negative_words':'The post include negative words',
        }

        # Step 2: define the balanced decision tree for this classification task
        tree = {
            'root': 'racist_words',
            'racist_words': {'yes': 'Yes', 'no': 'sexist_words'},
            'sexist_words': {'yes': 'Yes', 'no': 'swear_words'},
            'swear_words': {'yes': 'Yes', 'no': 'identity_words'},
            'identity_words': {'yes': 'negative_words', 'no': 'Maybe'},
            'negative_words': {'yes': 'Maybe', 'no': 'No'}
        }
    elif which_tree=='gpt3.5':
        # GPT3.5
        # Step 1: define criterions of the decision tree.
        criterions = {
            'contains_offensive_language':'This post contains offensive language',
            'intention_to_offend':'The intention of the post is to offend',
            'tone_neutral':'The tone of the post is neutral',
            'tone_ambiguous':'The tone of the post is ambiguous'
        }

        # Step 2: define the balanced decision tree for this classification task
        tree = {
            'root': 'contains_offensive_language',
            'contains_offensive_language': {'yes': 'intention_to_offend', 'no': 'tone_neutral'},
            'intention_to_offend': {'yes': 'Yes', 'no': 'Maybe'},
            'tone_neutral': {'yes': 'No', 'no': 'tone_ambiguous'},
            'tone_ambiguous': {'yes': 'Maybe', 'no': 'No'}
        }
    all_scores = {}
    for key in criterions.keys():
        sentence = criterions[key]
        score = get_entailment_score(sentence, model, tokenizer)
        all_scores[key] = score
    return criterions, tree, all_scores


def get_full_tree_social_speech():
    dataset = load_dataset('social_bias_frames')
    val_dataset = dataset["validation"]
    criterions, tree, all_score = get_bias_offense_tree()
    tot = 0
    acc = 0
    for item in tqdm(val_dataset):
        tot += 1
        sentence = item['post']

        label, val = tree_predict(sentence, criterions, tree, model, tokenizer, all_score, True)
        truth = item['offensiveYN']
        
        if truth == '1.0':
            truth = 'Yes'
        elif truth == '0.5':
            truth = 'Maybe'
        else:
            truth = 'No'

        if label == truth:
            acc += 1

        print(acc/tot)
            
        
       
    print(acc/tot)

def get_full_tree_social_speech_baseline():
    dataset = load_dataset('social_bias_frames')
    val_dataset = dataset["validation"]
    tot = 0
    acc = 0
    hate_tuple =  get_entailment_score('this is offensive ', model, tokenizer)
    nonhate_tuple = get_entailment_score('this is not offensive ', model, tokenizer)
    maybehate_tuple = get_entailment_score('this is maybe offensive ', model, tokenizer)
    for item in tqdm(val_dataset):
        tot += 1
        sentence = item['post']
        
        hatelabel, hateval = entailment('this is offensive ', sentence, model, tokenizer, hate_tuple, True)
        nohatelabel, nohateval = entailment('this is not offensive ', sentence, model, tokenizer, nonhate_tuple, True)
        maybehatelabel, maybehateval = entailment('this is maybe offensive ', sentence, model, tokenizer, maybehate_tuple, True)
        if hateval > nohateval and hateval > maybehateval:
            label = 'Yes'
        elif nohateval > hateval and nohateval > maybehateval:
            label = 'No'
        else:
            label = 'Maybe'
        truth = item['offensiveYN']
        if truth == '1.0':
            truth = 'Yes'
        elif truth == '0.5':
            truth = 'Maybe'
        else:
            truth = 'No'
        if label == truth:
            acc += 1
        
    print(acc/tot)
    
