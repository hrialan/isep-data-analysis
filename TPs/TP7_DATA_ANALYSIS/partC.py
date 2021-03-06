from textblob import TextBlob

Textblob_object = TextBlob("During the Iron Age, what is now metropolitan France was inhabited by the Gauls, a Celtic people. Rome annexed the area in 51 BC, holding it until the arrival of Germanic Franks in 476, who formed the Kingdom of Francia. The Treaty of Verdun of 843 partitioned Francia into East Francia, Middle Francia and West Francia. West Francia, which became the Kingdom of France in 987, emerged as a major European power in the Late Middle Ages, following its victory in the Hundred Years' War (1337-1453). During the Renaissance, French culture flourished and a global colonial empire was established, which by the 20th century would become the second largest in the world. The 16th century was dominated by religious civil wars between Catholics and Protestants (Huguenots). France became Europe's dominant cultural, political, and military power in the 17th century under Louis XIV. In the late 18th century, the French Revolution overthrew the absolute monarchy, establishing one of modern history's earliest republics and drafting the Declaration of the Rights of Man and of the Citizen, which expresses the nation's ideals to this day.");

words = Textblob_object.words
sentences = Textblob_object.sentences

for count, sentence in enumerate(sentences, start=1):
    print("Sentence " , count, ": " ,sentence.sentiment)


print(Textblob_object.translate(to='fr'))

