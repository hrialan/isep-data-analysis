import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import conll2000

filename = 'metamorphosis_clean.txt'
file=open(filename, 'rt')
text = file.read()
file.close()

sentences = sent_tokenize(text)
words = word_tokenize(sentences[0])
nltk.pos_tag(words)

text_words = text.split()
en_stops = set(stopwords.words('english'))

#remove stop words
for elt in en_stops:
    while elt in text_words:
        text_words.remove(elt)

#for i in range(6):
#    print(" ".join(conll2000.sents()[i]))

#for i in range(6):
#    print(conll2000.tagged_sents()[i])


#from gensim.summarization import summarize
#print(summarize(text))

from gensim.summarization import keywords
print(keywords(text))