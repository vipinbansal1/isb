import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from keras.preprocessing.text import Tokenizer as tokenizer
from nltk.stem import WordNetLemmatizer
import math
import numpy as np

df = pd.read_excel("name matching - list1.xlsx")

df.assignee = df.assignee.str.partition(' (')

df.assignee = df.sort_values("assignee")
df.assignee = df.assignee.str.replace(',','')
df.assignee = df.assignee.str.replace('.','')
df.assignee = df.assignee.str.replace('Corporation','Corp')
df.assignee = df.assignee.str.replace('Incorporated','Inc')
df.assignee = df.assignee.str.upper()
df.drop_duplicates(keep='first', inplace=True)

df.to_csv("cleanedIndustry.csv", index=False)

list2 = pd.read_excel("name matching task - list2.xlsx")
list2.conm = list2.conm.str.replace(',','')
list2.conm = list2.conm.str.replace('.','')
list2.to_csv("list2clean.csv", index=False)

df_concat = pd.concat([df.assignee,list2.conm]).to_frame()
df_concat.drop_duplicates(keep= 'first', inplace = True)
df_concat.to_csv("concat.csv", index=False)

'''           
df1 = pd.read_csv("withLabel.csv")

groupingBasedOnClustering = df1.groupby('Clustering')

#Listing Industry classification based upon actual label's for each cluster
clusterringOrder = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]*27
for key, item in groupingBasedOnClustering:
    dfForKey = groupingBasedOnClustering.get_group(key)
    groupingBasedOnActualLabel = dfForKey.groupby('Label')
    order = [0]*27
    for actualkey, actualitem in groupingBasedOnClustering:
        order[actualkey] = actualitem.shape[0]
    clusterringOrder[key] = order

#Identify the max value of each Label under each clustered group and marked as my accuracy
accuracy = [0]*27 
for i in range(27):
    #print(i)    
    loc = -1
    maxVal = -1
    indexOfCluster = -1
    for j in range(27):
        temp = np.max(clusterringOrder[j])
        if(temp>maxVal):
            maxVal = temp
            loc = clusterringOrder[j].index(maxVal)
            indexOfCluster = j
    for k in range(27):
        if(len(clusterringOrder[k])>1):
            clusterringOrder[k][loc] = 0 
    print("label:", loc, " cluster:", indexOfCluster, " maxval:", maxVal   )
    clusterringOrder[indexOfCluster] = [0]
    accuracy[loc] = maxVal   
print(accuracy)

df = pd.read_excel("company descriptions.xlsx")





for count, sentence in enumerate(df['company_description']):
        loc = 0
        maxWordCount = 0
        if (sentence != sentence):
            df.values[count][2] = df.values[count][1]
            


def cleaniningText(sentence):
    try:
        
        # Text cleaning
        sentence = re.sub("[^a-zA-Z]"," ",str(sentence))
        sentence = re.sub(r"<br />", " ", sentence)
        sentence = re.sub(r"   ", "", sentence)
        sentence = re.sub(r"  ", "", sentence)
        
        
        
        #spliting sentence
        split_sentence = sentence.lower().split()
        
       
        
        #removing stop words
        stopwordsList = set(stopwords.words("english"))
        after_removing_stopwords = [word for word in split_sentence if not word in stopwordsList]
        
        
        #Stemming
        after_stemming = []
        stemmer = SnowballStemmer('english')
        for word in after_removing_stopwords:
            stemmed_word = stemmer.stem(word)
            after_stemming.append(stemmed_word)
            
       
        #Lemmatizing
        lemmatizer = WordNetLemmatizer()
        lemmatized = [lemmatizer.lemmatize(word) for word in after_stemming]
        
    except Exception as e:   
        print(e)
    
    return " ".join(lemmatized)
'''



'''
cleanedData = []
try:
    for sentence in df['company_description']:
        out = cleaniningText(sentence)
        cleanedData.append(out)
except Exception as e:
    print(e)
print(out)


print(len(cleanedData))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(cleanedData)

kmeans = KMeans(n_clusters=27, max_iter=2)
kmeans_fit = kmeans.fit(tfidf)
'''


'''
df_industry = pd.read_excel("industry labels.xlsx")
stopwordsList = set(stopwords.words("english"))
industry_name = []
for industry in df_industry['industry']:
    industry_name_split = industry.lower().split()
    industry_name_split = [word for word in industry_name_split if not word in stopwordsList]
    industry_name.append(" ".join(industry_name_split))

labeling_data = []
try:
    for count, sentence in enumerate(df['company_description']):
        loc = 0
        maxWordCount = 0
        if (sentence != sentence):
            labeling_data.append(loc)
            continue
        split_sentence = sentence.lower().split()
        
        
        for index, industry_words in enumerate(industry_name):
            for (i, industry_word) in enumerate(industry_words.split()):
                currentWordCount = 0
                if(sentence.lower().find(industry_word) != -1):
                    currentWordCount = currentWordCount + 1
                if(currentWordCount > maxWordCount):
                    maxWordCount = currentWordCount 
                    loc = index
        labeling_data.append(loc)
except Exception as e:
    print(e,count)
df['Label'] = labeling_data
        
        #removing stop words
'''
        
        
  