# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 20:49:02 2016

@author: ogill
"""
##  Step 1 - Downloading the data  ## 
 
#scrapping data from "https://www.glassdoor.com.au/Reviews/Commonwealth-Bank-of-Australia-Reviews-E7922.htm"
d1  = 'Great work environment and fits in well with my family'
d2  = 'Good corporate culture that cares for the well being of its people'
d3  = 'Dynamic team, vast information management'
d4  = 'Good exposure to Banking domain, enterprise IT solutions. Mature SOA architecture capability in place.'
d5  = 'Has a number of ecosystems and culture climates. I was lucky in investment team to play hard but was rewarded well.'
d6  = 'Great culture and values. Love the work environment and flexibility'
d7  = 'Great company, technology, benefits and team. Customer leader in many areas including online banking :-).'
d8  = 'Friendly and supportive environment. Where you can find diferent and challenging projects to work on. Working in a flexible is possible and supported. At least in support functions.'
d9  = 'Great reputation and plenty of opportunities for training and diversification. Promotes a positive ethical culture within. Good work life balance for a corporate.'
d10 = 'Good progression and training options. Good social life. Good location of offices darling Park'

#appending all the scrapped data points into an array
data = []
for i in range(1,10):
    print 'd' + str(i)
    data.append(eval('d' + str(i)))
 
  
##  Step 2 - Importing the packages and initiating the classes for data preparation  ## 

#importing relevant libraries
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
#Initiating the word grabbing class
tokenizer = RegexpTokenizer('\w+')

#Initiating the instance to remove english stop words
stop = set(stopwords.words('english'))

#Initiating the word stemming class (to make long words shorter)
stem = PorterStemmer()


##  Step 3 - Preparing the data for Topic Modelling  ## 

#lowering and splitting words
clean_data = [i.lower() for i in data]
clean_data = [tokenizer.tokenize(i) for i in clean_data]

#removing stop words and making the remaing words shorter            
for i in range(1,len(clean_data)):
    clean_data[i] = [j for j in clean_data[i] if not j in stop]  
    clean_data[i] = [stem.stem(j) for j in clean_data[i]]


##  Step 4 - Piping the data into an LDA for Topic Modelling  ##   
  
#extracting the terms dictionary from the clean_data
clean_data_dict = corpora.Dictionary(clean_data)
    
#transforming the clean data into a term matrix
clean_data_corpus = [clean_data_dict.doc2bow(i) for i in clean_data]

#fitting a Latent Dirichlet Allocation model for topic modelling
topic_model = gensim.models.ldamodel.LdaModel(clean_data_corpus, num_topics=3, id2word = clean_data_dict, passes=3)

print(topic_model.show_topics())
