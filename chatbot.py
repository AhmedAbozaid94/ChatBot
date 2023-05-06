import tkinter as tk
from tkinter import scrolledtext
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import datetime

# load json file
with open("intents.json") as file:
    data = json.load(file)

# preprocessing data
stemmer = LancasterStemmer()
words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        # tokenize each word
        wrds = nltk.word_tokenize(pattern)
        wrds = [stemmer.stem(w.lower()) for w in wrds if w != "?"]
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
        
        # add the label to the labels list
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

# remove duplicates and sort
words = sorted(list(set(words)))
labels = sorted(labels)

# label encoding
training = []
output = []

out_empty = [0 for _ in range(len(labels))]
for x, doc in enumerate(docs_x):
    bag = []
    # stem each word
    wrds = [stemmer.stem(w) for w in doc]

    # create bag of words
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    # output is a 0 for each tag and 1 for current tag
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    # append to training and output lists
    training.append(bag)
    output.append(output_row)

# convert training and output lists to numpy arrays
training = np.array(training)
output = np.array(output)

# model building
model =tf.keras.models.Sequential([
    tf.keras.layers.Dense(8, input_shape=(len(training[0]),), activation='relu'),
    tf.keras.layers.Dense(8),
    tf.keras.layers.Dense(len(output[0]), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# training the model
model.fit(training, output, epochs=250, batch_size=32)

def bag_of_words(s):
    global words
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(w.lower()) for w in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)
#-------------------------
# get response from the bot
#-------------------------
def get_response(user_input):
    bag = bag_of_words(user_input)
    bag = np.reshape(bag, (1, len(bag)))
    results = model.predict(bag)
    results_index = np.argmax(results)
    tag = labels[results_index]

    responses = []
    for intent in data['intents']:
        if intent['tag'] == tag:
            responses = intent['responses']
            break
    
    if tag == 'time':
        now = datetime.datetime.now()
        time_str = now.strftime("%H:%M:%S")
        date_str = now.strftime("%A, %B %d, %Y")
        return f"The current time is {time_str} on {date_str}."
    elif len(responses) > 0:
        return random.choice(responses)
    else:
        return "I'm sorry, I don't understand."

#-------------------------
# GUI
#-------------------------
def send_message(event):
    user_input = user_input_box.get()
    response = get_response(user_input)
    chat_box.configure(state='normal')
    chat_box.insert('end', "You: " + user_input + "\n\n")
    chat_box.insert('end', "Bot: " + response + "\n\n")
    chat_box.configure(state='disabled')
    user_input_box.delete(0, 'end')

root = tk.Tk()
root.title("Chatbot")

# chat box
chat_box = scrolledtext.ScrolledText(root, width=80, height=20, state='disabled', font=("Arial", 16),background="light blue")
chat_box.grid(column=0, row=0, padx=10, pady=10)

# user input box
user_input_box = tk.Entry(root, width=50)
user_input_box.grid(column=0, row=1, padx=10, pady=10)

# bind the Enter key to the send_message function
user_input_box.bind("<Return>", send_message)

root.mainloop()
