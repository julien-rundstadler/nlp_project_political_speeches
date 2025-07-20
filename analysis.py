import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from scipy.special import softmax
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Data import

df = pd.read_csv("data/obama_speeches.csv")
df.head()

# Model loading

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')

# ----------------------- Sentiment extraction ---------------------

def extract_sentiment_distribution(text):
    try:
        sentences = [sent.text for sent in nlp(text).sents]
        sent_batch_size = 4
        batches = []
        for i in range(0, len(sentences), sent_batch_size):
            batch = " ".join(sentences[i:i+sent_batch_size])
            batches.append(batch)

        polarity = []
        subjectivity = []
        for batch in batches:
            polarity.append(nlp(batch)._.blob.polarity)
            subjectivity.append(nlp(batch)._.blob.subjectivity)  
        res_polarity = np.mean(polarity, axis=0)
        res_subjectivity = np.mean(subjectivity, axis=0)
        return res_polarity, res_subjectivity
    
    except Exception as ex:
        print("Error in Sentiment Extraction:")
        print(ex)
        print('Error was caught and we return neutral in this case.')
        return 2

# Adding polarity and subjectivity scores to dataframe

df['polarity_score'] = df['transcript'].apply(lambda x: extract_sentiment_distribution(x)[0])
df['subjectivity_score'] = df['transcript'].apply(lambda x: extract_sentiment_distribution(x)[1])
df.head()

# Date data correction

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date', ascending=True)
df['date'][18] = '2017-01-10'
df['date'][18]

# Plotting results

plt.style.use('ggplot')
plt.figure(figsize=(12, 4))

plt.subplot(2, 1, 1)
plt.plot(df['date'], df['polarity_score'], marker = '^', ms=5, color= 'red')
plt.title('Polarity', fontsize = 12)
plt.ylabel('Polarity')
plt.xlabel('Date of speech')

plt.subplot(2, 1, 2)
plt.plot(df['date'], df['subjectivity_score'], marker = 'o', ms=5, color = 'blue')
plt.title('Subjectivity', fontsize = 12)
plt.ylabel('Subjectivity')
plt.xlabel('Date of speech')

plt.tight_layout()
plt.show()

# ------------------- Named Entities Extraction -------------------

def extract_nes(text):
    try:
        dict = {"entity":[], "label":[]}
        sentences = [sent for sent in nlp(text).sents]
        for sentence in sentences:
            for ent in sentence.ents:
                dict['entity'].append(ent.text)
                dict['label'].append(ent.label_)
        return dict
    except Exception as ex:
        print("Error in NER extraction:")
        print(ex)
        return []
    
# Adding named entities for each speech to dataframe
df['named_entities'] = df['transcript'].apply(lambda x: extract_nes(x))
df.head()

# Gathering all the named entitities in one list
all_speeches_ner = []
count = 0
for d in df['named_entities']:
    count = count + 1
    all_speeches_ner = all_speeches_ner + d['entity']

print(count)
len(all_speeches_ner)

# Counting number of occurences
from collections import Counter

counter = Counter(all_speeches_ner)

# Get the 20 most frequent elements
most_common = counter.most_common(20)

# Get a dataframe with most frequent elements and their occurence
ner = pd.DataFrame(most_common)
ner.head()
ner = ner.iloc[::-1]

# Plotting results
plt.style.use('ggplot')
plt.barh(ner[0], ner[1])
plt.xlabel('Occurences')
plt.show()