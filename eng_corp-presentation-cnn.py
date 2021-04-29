#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
import numpy as np


# In[2]:


txt = pd.read_csv("eng_corp.csv")
txt


# In[3]:


#抽取cnn title, content
cnn = txt[txt['publication'].str.contains("CNN")]
cnn = cnn[['title','content']]
cnn
no_cnn = []
for i in cnn['content']:
    no_cnn.append(i.replace('(CNN)','').replace('Washington','').replace('Houston',''))
no_cnn
cnn['content'] = no_cnn


# # 主題分析

# In[4]:


from nltk.tokenize import WordPunctTokenizer
import nltk
from nltk.corpus import stopwords  
stop_words = set(stopwords.words('english'))
import string  
punt = []
for i in string.punctuation:
    punt.append(i)
topic = cnn['title']
topic = topic[~topic.str.contains('Fast Facts') & ~topic.str.contains('CNN 10')& ~topic.str.contains('CNN Student')]
len(topic)


# In[5]:


#POS
topic1 = []
for i in topic:
    topic1.append(nltk.word_tokenize(i))
topic2 = []
for i in topic1:
    topic2.append(nltk.pos_tag(i,tagset='universal'))


# In[6]:


#Count POS
stopword = ['’','s','“','”']
pos1 = {}
for i in topic2:
    for j in i:
        if j[1] != "." and j[1] != "X" and j[0] not in stopword:
            pos1[j[1]] = pos1.get(j[1],0) + 1
                        
df1 = pd.DataFrame.from_dict(pos1,  orient='index', columns=['count'])
df1 = df1.sort_values(by='count',ascending=False)
df1["mean"] = round(df1['count']/df1['count'].sum(),2)
df1


# In[7]:


#gerund
notGerund = ['everything', 'nothing', 'something', 'beijing']
gerundCount = 0
gerund = []
for i in topic2:
    for j in i:
        if j[1] == 'NOUN' and j[0].endswith('ing') and j[0].lower() not in notGerund:
            gerundCount += 1
            gerund.append(j[0])
print(gerundCount)
gerund


# In[8]:


topw = {}
for i in topic2:
    for j in i:
        if j[1] != "." and j[1] != "X" and j[0] not in stopword:
            topw[j[0]] = topw.get(j[0],0) + 1
topw = pd.DataFrame.from_dict(topw, orient='index', columns=['count'])
topw = topw.sort_values(by=['count'],ascending=False)
topw['mean'] = topw['count']/topw['count'].sum()
topw.head(10)


# In[9]:


#Total noun
noun = []
verb = []
for i in topic2:
    for j in i:
        if j[1] == 'NOUN':
            noun.append(j[0])
        if j[1] == 'VERB':
            verb.append(j[0])
print("Total noun:",len(noun))


# In[10]:


#Verb details
totv = len(verb)
print('Total verb:',totv)
ing = []
ed = []
for i in verb:
    ing.append(i.endswith('ing'))
    ed.append(i.endswith('ed'))
print('ing:',ing.count(True)/totv)
print('ed:',ed.count(True)/totv)
print('be',(verb.count('be')+(verb.count('is'))+verb.count('was')+verb.count('are')+verb.count('were'))/totv)

being = []
for i in topic2:
    for j in range(0, len(i)):
        if i[j][0][0].isupper() and i[j][0].endswith('ing'):
            continue
        elif i[j][0].endswith('ing') and i[j-1][1] == 'VERB':
            being.append(i[j])
print('Gerund Total:',ing.count(True)-len(being))


# In[11]:


#start word
start = []
start_pos = {}
for i in topic2:
    start.append(i[0])
for i in start:
    start_pos[i[1]] = start_pos.get(i[1],0) + 1
start_pos = pd.DataFrame.from_dict(start_pos,  orient='index', columns=['count'])
start_pos["mean"] = start_pos['count']/start_pos['count'].sum()
start_pos


# In[12]:


#Verbs in sentences
verb_w = []
for i in topic2:
    for j in i:
        if j[1] == 'VERB':
            verb_w.append(j[0])
verb_w2 = []
for i in verb_w:
    if i in verb_w2:
        continue
    else:
        verb_w2.append(i)
verb_w2


# In[13]:


noVerb = []
hasVerb = []

for i in topic:
    pos = nltk.pos_tag(nltk.word_tokenize(i.lower()),tagset='universal')
    verb = 0
    poshas = []
    posno = []
    for j in pos:
        if j[1] == 'VERB':
            verb += 1
    if verb == 0:
        noVerb.append(i)
    else:
        hasVerb.append(i)


# In[14]:


#Phrase before the first verb
top_count = []
start_sen = []
for i in topic:
    sen = i.split()
    top_count.append(len(sen))
    for j in sen:
        if j in verb_w2:
            pos = i.find(j)
            if pos != -1 or len(i[0:pos]) != 0:
                    start_sen.append(i[0:pos])
                    break
            
start_count = []
for i in start_sen:
    split = i.split()
    start_count.append(len(split))
print("proportion to topic:",len(start_sen)/len(topic))
start_sen


# In[15]:


#topics with verb
#word count
v_count = []
for i in hasVerb:
    sent = i.split()
    v_count.append(len(sent))
hasVerb


# In[16]:


#topics with no verb
noV_count = []
for i in noVerb:
    sent = i.split()
    noV_count.append(len(sent))
noVerb


# In[17]:


#Stat for phrase before the first verb
start_count = pd.DataFrame(start_count)
start_count.describe()


# In[18]:


#Stat for sentences
top_count = pd.DataFrame(top_count)
top_count.describe()


# In[19]:


#number of sentence without verb
(len(top_count)-len(start_count))/len(top_count)


# # 有動詞/無動詞主題分析

# In[20]:


#stat for no verb topic
noV_countDF = pd.DataFrame(noV_count)
noV_countDF.describe()


# In[21]:


#stat for verb-topic
V_countDF = pd.DataFrame(v_count)
V_countDF.describe()


# In[22]:


#no-verb/has-verb topics distribution details
nov_skew = pd.Series(noV_count)
print("No-verb topics skew", round(nov_skew.skew(),2))

hasv_skew = pd.Series(v_count)
print("Has-verb topics skew", round(hasv_skew.skew(),2))


import seaborn as sns
from matplotlib import pyplot as plt 
sns.kdeplot(noV_countDF[0],bw=1, label='Titles without verb dist.')
sns.kdeplot(V_countDF[0],bw=1, label='Titles with verb dist.')
plt.title('Word count distribution')


# In[23]:


def pos(sen, topicA, topicB):
    topic1 = []
    for i in sen:
        topic1.append(nltk.word_tokenize(i))
        
    for j in topic1:
        topic_clean = []
        pos = nltk.pos_tag(j,tagset='universal')
        for c in pos:
            if c[1] != '.' and c[0] != '’' and c[1] != 'X':
                topic_clean.append(c)
        topicA.append(topic_clean)
    
    for k in topicA:
        for n in k:
            if n[1] != '.' and n[0] != '’':
                topicB.append(n)


# In[24]:


#pos
nv_pos_e = []
nv_pos_a = []
pos(noVerb,nv_pos_e,nv_pos_a)
hasv_pos_e = []
hasv_pos_a = []
pos(hasVerb,hasv_pos_e,hasv_pos_a)


# In[25]:


#count no-verb topics
'''
Tag	Meaning	English Examples
ADJ	adjective	new, good, high, special, big, local
ADP	adposition	on, of, at, with, by, into, under
ADV	adverb	really, already, still, early, now
CONJ	conjunction	and, or, but, if, while, although
DET	determiner, article	the, a, some, most, every, no, which
NOUN	noun	year, home, costs, time, Africa
NUM	numeral	twenty-four, fourth, 1991, 14:24
PRT	particle	at, on, out, over per, that, up, with
PRON	pronoun	he, their, her, its, my, I, us
VERB	verb	is, say, told, given, playing, would
.	punctuation marks	. , ; !
X	other	ersatz, esprit, dunno, gr8, univeristy
'''

nov = {}
for i in nv_pos_a:
    nov[i[1]] = nov.get(i[1],0) + 1
novDF = pd.DataFrame.from_dict(nov,  orient='index', columns=['count'])
novDF = novDF.sort_values(by='count',ascending=False)
novDF["mean"] = round(novDF['count']/novDF['count'].sum(),2)
novDF


# In[26]:


#count has-verb topics
hasv = {}
for i in hasv_pos_a:
    hasv[i[1]] = hasv.get(i[1],0) + 1
hasvDF = pd.DataFrame.from_dict(hasv,  orient='index', columns=['count'])
hasvDF = hasvDF.sort_values(by='count',ascending=False)
hasvDF["mean"] = round(hasvDF['count']/hasvDF['count'].sum(),2)
hasvDF


# In[53]:


#no-verb topics analysis
lst = []
for i in range(2,15):
    length = {}
    for j in nv_pos_e:
        if len(j) == i:
            for k in j:
                length[k[1]] = length.get(k[1],0) + 1
    lst.append(length)
    
nv_each = pd.DataFrame(lst)
nv_each.index = np.arange(2, 15)

nv_each['noun'] = round(nv_each['NOUN']/nv_each.sum(axis=1),2)
nv_each['adj'] = round(nv_each['ADJ'].fillna(0)/nv_each.sum(axis=1),2)
nv_each['prep'] = round((nv_each['ADP'].fillna(0)+nv_each.fillna(0)['PRT'])/nv_each.sum(axis=1),2)
nv_each['det'] = round(nv_each['DET'].fillna(0)/nv_each.sum(axis=1),2)
nv_each = nv_each[['NOUN','DET','noun','adj','prep','det']]
nv_each = nv_each.reset_index()
nv_each


# In[54]:


#has-verb topics analysis
lst2 = []
for i in range(2,23):
    length = {}
    for j in hasv_pos_e:
        if len(j) == i:
            for k in j:
                length[k[1]] = length.get(k[1],0) + 1
    lst2.append(length)
    
hasv_each = pd.DataFrame(lst2)
hasv_each.index = np.arange(2, 23)

hasv_each['noun'] = round(hasv_each['NOUN']/hasv_each.sum(axis=1),2)
hasv_each['verb'] = round(hasv_each['VERB'].fillna(0)/hasv_each.sum(axis=1),2)
hasv_each['adj'] = round(hasv_each['ADJ'].fillna(0)/hasv_each.sum(axis=1),2)
hasv_each['prep'] = round((hasv_each['ADP'].fillna(0)+hasv_each['PRT']).fillna(0)/hasv_each.sum(axis=1),2)
hasv_each['det'] = round(hasv_each['DET'].fillna(0)/hasv_each.sum(axis=1),2)
hasv_each = hasv_each[['NOUN','DET','noun','verb','adj','prep','det']]
hasv_each = hasv_each.reset_index()
hasv_each


# In[29]:


#check the topics with n words
def check(topic, word):
    if topic == noVerb:
        print(nv_each.iloc[word-2],'\n')
    else:
        print(hasv_each.iloc[word-2],'\n')
        
    for i in topic:
        a = i.split()
        if len(a) == word:
            print(i)


# In[30]:


#noVerb: 2-14, hasVerb: 2-22
check(hasVerb,8)


# In[31]:


#find start word
def start(x,y):
    for i in x:
        if i.startswith(y):
            print(i)


# In[32]:


start(topic,'New')


# In[33]:


#find titles contain assigned words
def word(x):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth',None)
    print(topic[topic.str.contains(x)])


# In[34]:


word('making')


# # Title plots

# In[35]:


import seaborn as sns
from matplotlib.ticker import MaxNLocator


# In[36]:


#length:noun
sns.regplot(x='index', y='noun',data=nv_each,label='No verb')
sns.regplot(x='index', y='noun',data=hasv_each,label='Has verb')
plt.xlabel("Length")
plt.ylabel("Noun")
plt.title('Length:Noun',size=15)
plt.xticks(range(2,22,2))
plt.legend()


# In[37]:


#length:adjective
sns.regplot(x='index', y='adj',data=nv_each,label='No verb')
sns.regplot(x='index', y='adj',data=hasv_each,label='Has verb')
plt.xlabel("Length")
plt.ylabel("Adjective")
plt.title('Length:Adjective',size=15)
plt.xticks(range(2,22,2))
plt.legend()


# In[38]:


#length:preposition
sns.regplot(x='index', y='prep',data=nv_each,label='No verb')
sns.regplot(x='index', y='prep',data=hasv_each,label='Has verb')
plt.xlabel("Length")
plt.ylabel("Preposition")
plt.title('Length:Preposition',size=15)
plt.xticks(range(2,22,2))
plt.legend()


# In[63]:


#noun:adj
sns.regplot(x='adj', y='noun',data=nv_each,label='No verb')
sns.regplot(x='adj', y='noun',data=hasv_each,label='Has verb')
plt.xlabel("Adjective")
plt.ylabel("Noun")
plt.title('Adjective:Noun',size=15)
plt.legend()


# In[64]:


#noun:prep
sns.regplot(x='prep', y='noun',data=nv_each,label='No verb')
sns.regplot(x='prep', y='noun',data=hasv_each,label='Has verb')
plt.xlabel("Preposition")
plt.ylabel("Noun")
plt.title('Preposition:Noun',size=15)
plt.legend()


# In[58]:


#noun:det
sns.regplot(x='det', y='noun',data=nv_each,label='No verb')
sns.regplot(x='det', y='noun',data=hasv_each,label='Has verb')
plt.xlabel("Determiner")
plt.ylabel("Noun")
plt.title('Determiner:Noun',size=15)
plt.legend()


# # 情態分析

# In[43]:


from textblob import TextBlob


# In[44]:


#has-verb titles sentiment
nov_sen = []
for i in noVerb:
    blob = TextBlob(i)
    sen = blob.sentences[0].sentiment
    count = i.split()
    if sen[0] != 0:
        nov_sen.append([i, len(count), round(sen[0],2)])
    

#no-verb titles sentiment
hasv_sen = []
for i in hasVerb:
    blob = TextBlob(i)
    sen = blob.sentences[0].sentiment
    count = i.split()
    if sen[0] != 0:
        hasv_sen.append([i, len(count), round(sen[0],2)])


# In[45]:


#sentiment check

def senti(x, y, z=-1):
    for i in x:
        if i[1] == y and i[2] >= z:
            print(i[0],i[2])


# In[46]:


senti(nov_sen,4)


# # 名詞短語分析

# In[47]:


from textblob.np_extractors import ConllExtractor
extractor = ConllExtractor()


# In[48]:


nounp = []
for i in topic:
    blob = TextBlob(i, np_extractor=extractor)
    nph = blob.noun_phrases
    for j in nph:
        nounp.append(j)


# In[49]:


#np check
def npcheck(n):
    for i in nounp:
        count = i.split()
        if len(count) == n:
            print(i)


# In[50]:


npcheck(4)


# In[51]:


topic[topic.str.contains('Total lunar')]


# In[52]:




nltk.pos_tag(nltk.word_tokenize('are these photographs really 102 years old?'),tagset='universal')


# In[ ]:




