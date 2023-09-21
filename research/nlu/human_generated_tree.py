import openai
import backoff

prompt_head = """Write a Python function that constructs a decision tree according to the given examples that can generate the correct label of the given classification task.

### Available APIs(shared for all tasks):

\"\"\"Returns whether the hypothesis in entailed by the premise.\"\"\"
def entailment(hypothesis, premise, model, tokenizer):
    proposition = f'{hypothesis} is entailed by {premise}.'
    inputs = tokenizer(proposition,  return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)['logits'][0]
    ent_label = int(outputs[0] > outputs[2])
    if ent_label == 1:
        return 'yes'
    else:
        return 'no'

\"\"\"Use the constructed decision tree to predict the label of the sentence.\"\"\"
def tree_predict(sentence, criterions, tree, model, tokenizer):
    node = tree['root']
    while node not in POSSIBLE_CLASSES:
        ent_label = entailment(criterions[node], sentence, model, tokenizer)
        node = tree[node][ent_label]
    return node

### Task: Movie review classification
### Description: Determine if a movie review expresses positive attitude or negative attitude.
### Possible classes: [positive, negative]
### Examples:
- contains no wit, only labored gags
    - [The movie is wise|The movie is not wise|1], [the story is fun|the story is not boring|1], [the review is positive|the review is negative|1]
- that loves its characters and communicates something rather beautiful about human nature
    - [The characters are lovely|The characters are awful|0], [the script is touching|the script is dry|0], [the review is positive|the review is negative|0]
- on the worst revenge-of-the-nerds clichés the filmmakers could dredge up
    - [The movie is novel|The movie is mostly platitudes|1], [the review is negative|1]
- are more deeply thought through than in most right-thinking films
    - [The takeaway of the movie is profound|The idea of the movie is shallow|0], [the review is positive|the review is negative|0]

### Define possible classes
POSSIBLE_CLASSES = ['positive', 'negative']

### Decision Tree Logic:
Start by assessing whether the movie is interesting.
- If uninteresting:
    Probe for depth by determining if the movie is wise.
    - If wise, label as a positive review.
    - Otherwise, label as a negative review.
- If interesting:
    Examine the quality of the script.
    - If the script is commendable, label as a positive review.
    - If not a good script
        Evaluate the portrayal of characters.
        - If the characters are well-depicted, label the review as positive.
        - If characters are not good, label the review negative.

### Python program: 
\"""Decision tree for this task\"""
def get_decision_tree(sentence, model, tokenizer):
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

    return criterions, tree

"""

prompt_cola ="""
### Task: Grammar correctness classification
### Possible classes: [acceptable, unacceptable]

### Define possible classes
POSSIBLE_CLASSES = ['acceptable', 'unacceptable']

### Python program: 
\"""Decision tree for this task\"""
Try to generate a more balanced tree!
def get_decision_tree(sentence, model, tokenizer):

"""

def get_cola_tree():
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

    return criterions, tree

prompt_emotion ="""
### Task: Emotion classification
### Description: I wrote a sentence about my feeling, determine which emotion am I feeling.
### Possible classes: ['I feel sad', 'I feel happy', 'I feel love', 'I feel angry', 'I feel afraid', 'I feel surprised']

### Define possible classes
POSSIBLE_CLASSES = ['I feel sad', 'I feel happy', 'I feel love', 'I feel angry', 'I feel afraid', 'I feel surprised']

### Decision Tree Logic:

"""

def get_emotion_tree():
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
        'root': 'positive words',
        'positive_words': {'yes': 'intensifiers', 'no': 'fear'},
        'intensifiers': {'yes': 'unexpected', 'no': 'I feel happy'},
        'unexpected': {'yes': 'I feel surprised', 'no': 'I feel love'},
        'fear': {'yes': 'I feel afraid', 'no': 'upset'},
        'upset': {'yes': 'I feel sad', 'no': 'I feel angry'}
    }

    return criterions, tree

prompt_amazon = """
### Task: Amazon Review Star classification
### Description: Given an amazon review, determine the star(1-5) of this review, where 1 is the worst and 5 is the best.
### Possible classes: ['1', '2', '3', '4', '5']

### Define possible classes
POSSIBLE_CLASSES = ['1', '2', '3', '4', '5']

### Python program: 
\"""Decision tree for this task\"""
Try to generate a balanced tree!
def get_decision_tree(sentence, model, tokenizer):

"""

def get_amazon_tree():
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

    return criterions, tree

prompt_hate_speech = """
### Task: Bias/Hateful speech classification
### Description: Given a speech from a white supremacy forum, determine if this is speech contains hate (eg. bias/racisim/prejudice...) towards a certain group of people.
### Possible classes: ['this is hate', 'this is noHate']

### Define possible classes
POSSIBLE_CLASSES = ['this is hate', 'this is noHate']

### Each criterion should be simple, clear, and concise.
### Start each criterion with 'This is'
### Decision Tree Logic:

"""

def get_hate_tree():
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

    return criterions, tree

prompt_social_bias_offensive = """
### Task: Offensive classification
### Description: Given a post, determine if this post is offensive.
### Possible classes: ['Yes','Maybe', 'No']

### Define possible classes
POSSIBLE_CLASSES = ['Yes','Maybe', 'No']

### Each criterion should be simple, clear, and concise.
### Start each criterion with 'This is'
### Generate at least 5 criterions!
### Decision Tree Logic:
"""

def get_sbic_tree():
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

    return criterions, tree

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

model_name = 'gpt-3.5-turbo'
content = prompt_head + prompt_cola
response = completions_with_backoff(
    model=model_name, 
    messages=[
        {"role": "user","content": content}
        ],
    )
response = response["choices"][0]["message"]["content"]
print(response)
