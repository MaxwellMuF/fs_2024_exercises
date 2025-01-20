import os
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

# Get and print the current working directory
current_working_directory = os.getcwd()
print(f"The current working directory is: {current_working_directory}")

# Change the working directory to a new directory (replace with the path you want)
new_working_directory = "Sub_7_Sentiment_Analyzer_BERT"
os.chdir(new_working_directory)

# Get and print the new current working directory
new_current_working_directory = os.getcwd()
print(f"The new current working directory is: {new_current_working_directory}")


#  ----------------------------------------------------------------------------
from transformers import pipeline

# load trained, serialized model
newmodel = pipeline('text-classification', model='my_saved_model', device=0)

prompt_list = ['This movie is great!', 'This movie sucks', 'This movie is not bad', 'This movie is not that bad',
               'I want to see it again', 'I don\'t want to see it again']

for prompt in prompt_list:
    print(f"\nPrompt: {prompt}")
    print(f"Model prediction: {newmodel(prompt)}")

# # Sentiment: positive
# newmodel('This movie is great!')

# # Sentiment: negative
# newmodel('This movie sucks')

# # Sentiment: positive
# newmodel('This movie is not bad')

# # Sentiment: positive
# newmodel('This movie is not that bad')

# # Sentiment: positive
# newmodel('I want to see it again')

# # Sentiment: negative
# newmodel('I don\'t want to see it again')


