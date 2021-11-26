import nltk
from nltk.corpus import wordnet as wn
import numpy as np

print(wn.synsets("Love"))

print(wn.synset('love.n.01').definition())
print(wn.synset('love.n.01').lemmas())
print(wn.synset('love.n.01').examples())



antonyms = []
synonyms = []
for syn in wn.synsets("Love"):
    for lem in syn.lemmas():
        synonyms.append(lem.name())
        if lem.antonyms():
            antonyms.append(lem.antonyms()[0].name())

print("Synonyms of Love are ",np.unique(synonyms))
print("Antonyms of Love are " ,np.unique(antonyms))

print(sorted(wn.langs()))

print(wn.synset('love.n.01').lemma_names('jpn'))

cat = wn.synset('cat.n.01')
dog = wn.synset('dog.n.01')
car = wn.synset('car.n.01')

#path_similarity
print()
print('path_similarity')
print('dog/cat: ',dog.path_similarity(cat))
print('cat/car: ',cat.path_similarity(car))
print('dog/car: ',dog.path_similarity(car))


#lch_similarity
print()
print('lch_similarity')
print('dog/cat: ',dog.lch_similarity(cat))
print('cat/car: ',cat.lch_similarity(car))
print('dog/car: ',dog.lch_similarity(car))

#wup_similarity
print()
print('wup_similarity')
print('dog/cat: ',dog.wup_similarity(cat))
print('cat/car: ',cat.wup_similarity(car))
print('dog/car: ',dog.wup_similarity(car))