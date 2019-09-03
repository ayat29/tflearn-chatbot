import nltk
import numpy
import tensorflow
import random
import json
from nltk.stem.lancaster import LancasterStemmer
root = LancasterStemmer()
import tflearn
import pickle

with open("chatdata.json") as file:
    source = json.load(file)
buckets = []
words = []
ordered_buckets = []
ordered_words = []
for bucket in source["buckets"]:
    for prompt in bucket["prompts"]:
        tokenized = nltk.word_tokenize(prompt)
        words.extend(tokenized)
        ordered_words.append(tokenized)
        ordered_buckets.append(bucket["type"])
    if bucket["type"] not in buckets:
        buckets.append(bucket["type"])
words = [root.stem(a.lower()) for a in words if a != "?"]
words = sorted(list(set(words)))
buckets = sorted(buckets)
input_d = []
output = []
output_raw = [0 for _ in range(len(buckets))]
for x, w in enumerate(ordered_words):
    pot = []
    stemmed = [root.stem(a.lower()) for a in w]
    for w in words:
        if w in stemmed:
            pot.append(1)
        else:
            pot.append(0)
    output_h = output_raw[:]
    output_h[buckets.index(ordered_buckets[x])] = 1
    input_d.append(pot)
    output.append(output_h)
input_d = numpy.array(input_d)
output = numpy.array(output)

tensorflow.reset_default_graph()
net = tflearn.input_data(shape=[None, len(input_d[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation = "softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)
model.fit(input_d, output, n_epoch=500, batch_size=4, show_metric=True)
model.save("model.tflearn")
def converter(sentence, words):
    pot = [0 for _ in range(len(words))]
    tokenized = nltk.word_tokenize(sentence)
    stemmed = [root.stem(w.lower()) for w in tokenized]
    for c in stemmed:
        for i, st in enumerate(words):
            if st == c:
                pot[i] = 1
    return numpy.array(pot)
def talk():
    print("Hey!")
    while True:
        say = input("You: ")
        if say.lower() == "quit":
            break
        talkback = model.predict([converter(say, words)])
        return_index = numpy.argmax(talkback)
        response = buckets[return_index]
        for t in source["buckets"]:
            if t["type"] == response:
                back =  t["returns"]
        print(random.choice(back))
        print(talkback)
talk()
