
# In very short summary, the below uses an LLM to convert text to numbers, and then those numbers are compared using similarity search of Faiss. A 'lucky_person' is compared to a number of 'people' in a list. Essentially it returns a sorted list of nearest neighbors in vector space. It eventually asks an LLM what the 2 people could talk about. 

# Prerequisites: install python, install ollama, "pip install packages requests numpy faiss-cpu"

import requests # Explain Faiss: comparing vectors --> Similarity searches
import numpy as np # numpy allows to work with arrays in Python
import faiss # Explain Faiss: comparing vectors --> Similarity searches. This is the engine used to actually compare "strings"

d = 4096 # Dimensionality, has to be in line with the model you use LLama3 uses 4096

# Input for Vector conversion
people = [ 
    'experience: 4 years, hobbies: walking in the park, building fancy front-end applications, expertise in: rocket science, visiting home country india, spending time with family', # 1 Jagrati
    'experience: 10 years, hobbies: working out, kite surfing, enjoying proper holidays, expertise in: delivery management, financial industry, offshore delivery', # 2 Youri
    'Experience: 14 years of professional wizardry Hobbies: Painting (the Picasso kind) and cooking (MasterChef level) Expertise in: Solving puzzles faster than you can say Rubiks Cube', # 3 Madhuri
    'experience: 1y, hobbies: yoga, cooking, traveling, expertise in: python/pyspark, data science, data analysis, a bit of devops', # 4 Laura
    'experience: 13 years, hobbies: video games, reading books, expertise in: Test automation, AWS, Pega', # 5 PK 
    'experience: 2 years, hobbies: dance, drawing, expertise in: sustainability', # 6 Nate
    'experience: 10 years, hobbies: playing piano, fooling around with large language models, generating art with stable diffusion, playing old fashioned video games, expertise in: project delivery, coding in python, innovation' # 7 Wynand
]

index = faiss.IndexFlatL2(d) # Could consider this your to be filled DB of vectors

X = np.zeros((len(people), d), dtype='float32') #initiates an empty array of 4096 'dimensions' (/numbers)

print("Value of X initiated with zeroes: \n", X) # Essentially created 5 placeholders for embeddings of each person

for i, person in enumerate(people):
    res = requests.post('http://localhost:11434/api/embeddings', 
                    json={
                        'model': 'llama3',
                        'prompt': person
                    })
    embedding = res.json()['embedding']
    X[i] = np.array(embedding)
    print("Value of X replacing zeroes: \n", X, end='\r')
    # All this does, is take each person, convert it to an embedding, and store it (overwriting the zero's)

index.add(X) # add it to your list of 'searchable items'

lucky_person = 'experience: 15 years, hobbies: geocaching, fiddling around with outdated hardware, building dockerswarms, watching star wars, play with lego, making midi music, expertise in technology consulting, delegating everything away' # Harm-Jan

res = requests.post('http://localhost:11434/api/embeddings', 
                json={
                    'model': 'llama3',
                    'prompt': lucky_person
                })
embedding = np.array([res.json()['embedding']], dtype='float32') # convert your search into yet another embedding

print("\nMy search query converted into an embedding: \n", embedding)

D, I = index.search(embedding, len(people)) # search 'nearest' vectors compared to lucky_person

print("Ordered list of array indexes: ", I) # -> shows order of array to display
print("Distances of ordered array items in high dimensional space: ", D) # -> shows nearness of embeddings after ordering
print(np.array(people)[I.flatten()]) # print a sorted list from most comparable to least comparable

selected = np.array(people)[I.flatten()]


# Building the RAG - retrieval augmented generation Prompt
augmented_generation_prompt = "\n\nwhat should the following 2 people talk about in a first get to know you session? \nPerson 1: " + lucky_person + "\nPerson 2: " + selected[0] +"\nAnswer with a prioritized list of topics"

print(augmented_generation_prompt)

res = requests.post('http://localhost:11434/api/generate', 
                json={
                    'model': 'llama3',
                    'prompt': augmented_generation_prompt,
                    'stream': False
                })
response = np.array([res.json()['response']])

print("\n\na first conversation should go about: ", response)