import openai
import backoff
openai.api_key_path = './api_key.txt'
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

prompt_head2 = """
Write a Python function that constructs a decision tree according to the given examples that can generate the correct label of the given classification task.

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

### Task: Grammar correctness classification
### Possible classes: ['accpetable', 'unacceptable']

### Define possible classes
POSSIBLE_CLASSES = ['accpetable', 'unacceptable']

### Decision Tree Logic:
- If verbs are not correctly constructed, the sentence is immediately labeled as unacceptable.
- If verbs are correct:
    The tree then checks if the sentence has correct punctuation
    - If incorrect, label the sentence as unacceptable
    - If correct:
        The next criterion to be assessed is the subject-verb agreement.
        - If subject and verb disagree, label the sentence as unacceptable.
        - If they agree, check for sentence fragments.
            - If the sentence is a fragment, label it as unacceptable.
            - If it is not a sentence fragment, label the sentence as acceptable.

### Python code for the decision tree:

```python
def get_decision_tree(sentence, model, tokenizer):
    # Step 1: define criterions of the decision tree
    criterions = {
        'correct_verbs': 'The verbs are correctly constructed in the sentence',
        'correct_punctuation': 'The sentence is punctuated correctly',
        'subject_verb_agreement': 'The subject and verb agree in the sentence',
        'no_sentence_fragments': 'The sentence is not a fragment',
    }

    # Step 2: define the balanced decision tree for this classification task
    tree = {
        'root': 'correct_verbs',
        'correct_verbs': {'yes': 'correct_punctuation', 'no': 'unacceptable'},
        'correct_punctuation': {'yes': 'subject_verb_agreement', 'no': 'unacceptable'},
        'subject_verb_agreement': {'yes': 'no_sentence_fragments', 'no': 'unacceptable'},
        'no_sentence_fragments': {'yes': 'acceptable', 'no': 'unacceptable'}
    }

    return criterions, tree
```

"""

prompt_sst2 = """
### Task: Movie review classification
### Possible classes: [positive, negative]

### Define possible classes
POSSIBLE_CLASSES = [positive, negative]

### Decision Tree Logic:

"""
prompt_cola ="""
### Task: Grammar correctness classification
### Possible classes: [acceptable, unacceptable]

### Define possible classes
POSSIBLE_CLASSES = ['acceptable', 'unacceptable']

### Decision Tree Logic:

"""

prompt_qnli ="""
### Task: the sentence answers the question classification
### Description: Determine if the sentence is an answer to the question
### Possible classes: [true, false]

### Define possible classes
POSSIBLE_CLASSES = ['true', 'false']

### Python program: 
\"""Decision tree for this task\"""
Try to generate a balanced tree!
def get_decision_tree(sentence, model, tokenizer):

"""

prompt_emotion_separate ="""
### Task: Sadness classification
### Description: Assume that "I" wrote a sentence, determine if "I" am feeling sad.
### Possible classes: ['Yes', 'No']

### Define possible classes
POSSIBLE_CLASSES = ['Yes', 'No']

### Python program: 
\"""Decision tree for this task\"""
Try to generate a balanced tree!
def get_decision_tree(sentence, model, tokenizer):

"""

prompt_emotion ="""
### Task: Emotion classification
### Description: I wrote a sentence about my feeling, determine which emotion am I feeling.
### Possible classes: ['I feel sad', 'I feel happy', 'I feel love', 'I feel angry', 'I feel afraid', 'I feel surprised']

### Define possible classes
POSSIBLE_CLASSES = ['I feel sad', 'I feel happy', 'I feel love', 'I feel angry', 'I feel afraid', 'I feel surprised']

### Decision Tree Logic:

"""
prompt_amazon = """
### Task: Amazon Review Star classification
### Description: Given an amazon review, determine the star(1-5) of this review, where 1 is the worst and 5 is the best.
### Possible classes: ['1', '2', '3', '4', '5']

### Define possible classes
POSSIBLE_CLASSES = ['1', '2', '3', '4', '5']

### Decision Tree Logic:

"""

prompt_agnews = """
### Task: News content classification
### Description: Given a piece of news, determine which category it belongs to.
### Possible classes: ['this is world news', 'this is sport news', 'this is business news', 'this is technology news']


### Define possible classes
POSSIBLE_CLASSES = ['this is world news', 'this is sport news', 'this is business news', 'this is technology news']

### Start each criterion with "This is"
### Each criterion should be general/high-level! 
### Python program: 
\"""Decision tree for this task\"""
Try to generate a balanced tree!
def get_decision_tree(sentence, model, tokenizer):

"""

prompt_agnews_separate = """
### Task: Science news classification
### Description: Given a piece of news, determine if this is a science news.
### Possible classes: ['Yes', 'No']

### Define possible classes
POSSIBLE_CLASSES = ['Yes', 'No']

### Decision Tree Logic:

"""

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
prompt_social_bias_offensive_deberta = """
### Task: Offensive classification
### Description: Given a post, determine if this post is potentially offensive to anyone.
### Possible classes: ['this is offensive','this is not offensive', 'this is maybe offensive']

### Define possible classes
POSSIBLE_CLASSES = ['this is offensive','this is not offensive', 'this is maybe offensive']

### Each criterion should be simple, clear, and concise.
### Start each criterion with 'This is'
### Decision Tree Logic:
"""
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

model_name = 'gpt-4'
# To generate a tree for SST2, use prompt_head2 instead of prompt_head
content = prompt_head + prompt_social_bias_offensive
response = completions_with_backoff(
    model=model_name, 
    messages=[
        {"role": "user","content": content}
        ],
    )
response = response["choices"][0]["message"]["content"]
print(response)