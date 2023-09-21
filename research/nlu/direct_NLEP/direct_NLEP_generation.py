import openai
import backoff

prompt_head = """Write a bug-free Python program that can generate the answer to the given instruction when correctly executed. Do not ask for user input. For reasoning tasks, define functions first and then define variables. For knowledge intensive tasks, define variables before defining functions.

### Instruction: Discuss the causes of the Great Depression
### Input: None
### Python program:

# Step 1: Import necessary built-in ligraries
# No need to import

# Step 2: Define dictionaries storing the knowledge about the grat depression
depression_name = "The Great Depression"
depression_period = "1929-1939"
depression_countries = "the United States and countries around the world"
depression_causes = {
    "Stock Market Crash of 1929": "In October of 1929, the stock market experienced a significant fall that wiped out millions of investors. This event is considered by many to be the initial trigger of the Great Depression.",
    "Overproduction": "During the 1920s, many industries produced more goods than consumers wanted or could afford. This ultimately led to a decline in demand for goods, causing job loss, lower wages, and business failure.",
    "High Tariffs and War Debts": "Protectionist trade policies in the form of high tariffs led to a decline in global trade, as other countries retaliated with tariffs of their own. Additionally, many countries were struggling to repay war debts, which led to economic instability.",
    "Bank Failures": "As demand for goods declined, many banks began to fail, causing a loss of confidence in the banking system. This led to a massive withdrawal of money from banks, causing even more banks to fail.",
    "Drought Conditions": "The Dust Bowl was a severe drought and dust storm that hit the Great Plains region of the United States in the 1930s. This had a significant impact on agriculture, causing many farmers to lose their land and livelihoods which worsened the effects of the depression."
}

# Step 3: Define necessary functions that generally solve this type of problem
# Do not need to define functions

# Step 4: Print an answer in natural language using the knowledge defined above
print(f"{depression_name} was a period of economic decline that lasted from {depression_period}, making it the longest-lasting depression in modern history. It affected not only {depression_countries}, causing substantial social and economic upheaval.\n")
print(f"There were several major causes of {depression_name}, which include:\n")
for i, (cause, description) in enumerate(depression_causes.items(), 1):
    print(f"{i}. {cause} - {description}\n")
print(f"Overall, {depression_name} was caused by a combination of factors, including economic, environmental, and political factors. Its impact was widespread, affecting millions of people around the world.")
```

### Instruction: Identify the odd one out.
### Input: Twitter, Instagram, Telegram
### Python program:
```
# Step 1: Import necessary built-in libraries
from collections import OrderedDict

# Step 2: Define dictionaries storing the knowledge about the main function of each application
services = {
    "Twitter": "a social media platform mainly for sharing information, images and videos",
    "Instagram": "a social media platform mainly for sharing information, images and videos",
    "Telegram": "a cloud-based instant messaging and voice-over-IP service",
}

# Step 3: Define a function that finds the different application
def find_odd_one_out(services, input_services):
    descriptions = [services[service] for service in input_services]
    for description in descriptions:
        if descriptions.count(description) == 1:
            return input_services[descriptions.index(description)]
    return None

# Step 4: Print an answer in natural language using the knowledge and function defined above
input_services = ["Twitter", "Instagram", "Telegram"]
odd_one_out = find_odd_one_out(services, input_services)
if odd_one_out:
    other_services = [service for service in input_services if service != odd_one_out]
    print(f"The odd one out is {odd_one_out}. {other_services[0]} and {other_services[1]} are {services[other_services[0]]} while {odd_one_out} is {services[odd_one_out]}.")
```

### Instruction: Calculate the total surface area of a cube with a side length of 5 cm.
### Input:  None
### Python program:
```
# Step 1: Import necessary built-in libraries
# No need to import

# Step 2: Define a function that calculate the surface area of cubes
def calculate_surface_area(side_length):
    return 6 * (side_length ** 2)

# Step 3: Define dictionaries storing the cube information
cube = {
    "side_length": 5  # Side length of the cube
}

# Step 4: Print a step-by-step calculation answer in natural language using the defined function and varible
side_length = cube["side_length"]
surface_area = calculate_surface_area(side_length)
print(f"The surface area of a cube is found by calculating the area of one of its faces and multiplying it by six (since a cube has six faces). The area of a cube face is simply its side length squared.\n")
print(f"Thus for this particular cube:")
print(f"Surface Area = 6 × (Side Length)²")
print(f"             = 6 × ({side_length} cm)²")
print(f"             = 6 × {side_length**2} cm²")
print(f"             = {surface_area} cm²\n")
print(f"The total surface area of this cube is {surface_area} square centimeters.")
"""
prompt_sst2 = """

### Instruction: Given a movie review, determine the attitude of this review as either positive or negative.
### Input: A moview review.
### Python program:

"""
prompt_cola = """

### Instruction: Given a sentence, determine the grammar correctness of this sentence as either acceptable or unacceptable.
### Input: A sentence.
### Python program:

"""
prompt_emotion = """

### Instruction: Given a sentence, classify the emotion of the sentence as one of these classes:  ['sad', 'happy', 'love', 'angry', 'afraid', 'surprised']. Your choice should be strictly the name of one of these 6 classes.
### Input: A sentence "I feel xxx"
### Python program:

"""
prompt_amazon = """

### Instruction: Given an amazon review, determine the star(1-5) of this review, where 1 is the worst and 5 is the best.
Your choice should be strictly in [1,2,3,4,5]
### Input: An amazon review.
### Python program:

"""

prompt_hsd = """

### Instruction: Given a speech from a white supremacy forum, determine if this is speech contains hate (eg. bias/racisim/prejudice...) towards a certain group of people.
### Input: A speech from a forum.
### Python program:

"""

prompt_sbic = """

### Instruction: Given a post, determine if the content of the post is offensive, maybe offensive, or not offensive. Classify the post as: ['Yes', 'Maybe', 'No']
### Input: A potentially offensive post.
### Python program:

"""
prompt_agnews = """

### Instruction: Given a piece of news, determine which category it belongs to. Classify it as one of the four classes: ['world news', 'sport news', 'business news', 'technology news']
### Input: A piece of news.
### Python program:

"""
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

model_name = 'gpt-4'
content = prompt_head + prompt_agnews
response = completions_with_backoff(
    model=model_name, 
    messages=[
        {"role": "user","content": content}
        ],
    )
response = response["choices"][0]["message"]["content"]
print(response)






