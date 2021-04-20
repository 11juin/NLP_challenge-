import pandas as pd  
from pdb import set_trace
from collections import Counter
from itertools import compress
import numpy as np
from adjustText import adjust_text
import os
from keras.models import Sequential
from keras import layers
import sklearn.metrics as metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Utils.utils import process_titles,preprocess,build_freqs,filter_docs,document_vector,has_vector_representation,extract_features
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
import nltk
#nltk.download('punkt')
from sklearn.model_selection import train_test_split
import string
import operator
import gensim
from sklearn.manifold import TSNE
from adjustText import adjust_text
import datetime

# Plot most popular words per year
def Most_used_words(data):
    for i in pd.DatetimeIndex(data['date_created']).year.unique().tolist():
        data_ = data.iloc[pd.DatetimeIndex(data['date_created']).year == i]
        for j in pd.DatetimeIndex(data_['date_created']).month.unique().tolist():
            sentences_ = data_.iloc[pd.DatetimeIndex(data_['date_created']).month == j].title.str.cat(sep = '')
            tok_ = word_tokenize(sentences_)
            tok_ = [char for char in tok_ if char not in string.punctuation]
            stop = stopwords.words('english')
            stop.extend(['the','says','new','first','said','group','may','per'])
            tok_ = [token.lower() for token in tok_]
            tok_ = [token for token in tok_ if token not in stop]
            tok_ = [token for token in tok_ if  token.isalpha()]
            tok_ = [word for word in tok_ if len(word) >= 3]
            tok_ = [tokk.capitalize() for tokk in tok_]
            plt.figure(figsize=(10,10))
            wc = WordCloud(max_font_size=40, max_words=100, background_color='white')
            wordcloud_ = wc.generate_from_text(' '.join(tok_))
            plt.imshow(wordcloud_, interpolation='bilinear')
            plt.axis('off')
            month = datetime.date(1900, j, 1).strftime('%B')
            label =  'Popular words on ' + str(month) + ' ' + str(i)
            plt.title(label, color = 'navy', size = 25)
            temp_folder = 'Images/MostPopularWords/' + str(i) +'/'
            if not os.path.exists(temp_folder):  os.makedirs(temp_folder)
            temp_name = temp_folder + str(month)
            plt.savefig(temp_name + '.png')
            freq_dis_={}
            for tok in tok_:
                if tok in freq_dis_:
                    freq_dis_[tok]+= 1
                else:
                    freq_dis_[tok]=1
            sorted_freq_ = sorted(freq_dis_.items(), key=operator.itemgetter(1), reverse=True)
            plt.figure(figsize=(10, 5))
            Freq_dist_nltk=nltk.FreqDist(tok_)
            Freq_dist_nltk.plot(50, cumulative=False)
            label =  'Popular words on ' + str(month) + ' ' + str(i)
            plt.title(label, color = 'navy', size = 25)
            plt.savefig(temp_name + '_bar.png')
            plt.close()

def Google_embedding_analysis(data):
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True) 
    stop_words = set(stopwords.words('english'))
    for i in pd.DatetimeIndex(data['date_created']).year.unique().tolist():
        data_ = data.iloc[pd.DatetimeIndex(data['date_created']).year == i]
        data_ =  data_.loc[data_.up_votes>= np.percentile(data_.up_votes, 95)]
        titles_list = [title for title in  data_['title']]
        big_title_string = ' '.join(titles_list)
        tokens = word_tokenize(big_title_string)
        words = [word.lower() for word in tokens if word.isalpha()]
        words = [word for word in words if not word in stop_words]
        vector_list = [model[word] for word in words if word in list(model.key_to_index.keys()) ]
        words_filtered = [word for word in words if word in list(model.key_to_index.keys()) ]
        word_vec_zip = zip(words_filtered, vector_list)
        word_vec_dict = dict(word_vec_zip)
        df = pd.DataFrame.from_dict(word_vec_dict, orient='index')
        tsne = TSNE(n_components = 2, init = 'random', random_state = 10, perplexity = 100)
        tsne_df = tsne.fit_transform(df)
        sns.set()
        fig, ax = plt.subplots(figsize = (11.7, 8.27))
        sns.scatterplot(tsne_df[:, 0], tsne_df[:, 1], alpha = 0.5)
        texts = []
        words_to_plot = list(np.arange(0, 400, 10))
        for word in words_to_plot:
            texts.append(plt.text(tsne_df[word, 0], tsne_df[word, 1], df.index[word], fontsize = 25))
        adjust_text(texts, force_points = 0.4, force_text = 0.4, 
                    expand_points = (2,1), expand_text = (1,2),
                    arrowprops = dict(arrowstyle = "-", color = 'black', lw = 0.5))
        temp_folder = 'Images/TopVotesWordsSimilarity/'+ str(i) +'/'
        if not os.path.exists(temp_folder):  os.makedirs(temp_folder)
        plt.savefig(temp_folder + str(i)+'.png')
        #plt.show()
        


def google_embedding_titles(data):
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True) 
    for ii in pd.DatetimeIndex(data['date_created']).year.unique().tolist():
        data_ = data.iloc[pd.DatetimeIndex(data['date_created']).year == ii]
        data_high = data_.sort_values('up_votes',ascending = False).head(30)
        data_low = data_.sort_values('up_votes',ascending = True).head(30)
        titles_list = [title for title in data_high['title']] + [title for title in data_low['title']]
        corpus = [preprocess(title) for title in titles_list]
        corpus, titles_list  = filter_docs(corpus, titles_list, lambda doc: has_vector_representation(model, doc))
        corpus, titles_list = filter_docs(corpus, titles_list, lambda doc: (len(doc) != 0))
        x = []
        for doc in corpus: 
            x.append(document_vector(model, doc))
        X = np.array(x) 
        tsne = TSNE(n_components = 2, init = 'random', random_state = 10, perplexity = 100)
        tsne_df = tsne.fit_transform(X)
        fig, ax = plt.subplots(figsize = (100, 100))
        sns.scatterplot(tsne_df[:, 0], tsne_df[:, 1], alpha = 0.5)
        texts = []
        titles_to_plot  = np.arange(0, len(titles_list), 1).tolist()
        for (j,title) in zip(range(len(titles_to_plot)),titles_to_plot):
            if j <= len(titles_to_plot)/2:
                i='navy'
            else:
                i='orange'
            texts.append(plt.text(tsne_df[title, 0], tsne_df[title, 1], titles_list[title], fontsize = 10,color = i))
        adjust_text(texts, force_points = 0.4, force_text = 0.4, 
                    expand_points = (2,1), expand_text = (1,2),
                    arrowprops = dict(arrowstyle = "-", color = 'grey', lw = 0.5))
        text = 'Blue:Top 99% Up Votes ; Orange: Below 1% Up Votes'
        plt.text(-50,100,s=text,size=15, color='black')
        temp_folder = 'Images/' + 'TopBottomVotesTitleSimilarity/'+ str(ii) +'/'
        if not os.path.exists(temp_folder):  os.makedirs(temp_folder)
        plt.savefig(temp_folder + 'TitleSimilarityOnVotes.png')
        plt.show()



def title_vote_prediction(data, benchmark = {'LogisticRegression':True,'NaiveBayes':True}, model = {'NN':True}):
    model_ = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True) 
    data['if_high_vote'] =  np.where(list(data.up_votes) <= data.up_votes.median(), 0, 1)
    Y = data['if_high_vote'].tolist()
    X_train, X_test, y_train, y_test = train_test_split(data['title'].tolist(), Y, test_size=0.2, random_state=42)
    # Generate Google embedding features: x_train/x_test
    x_train = []
    for doc in X_train: 
        x_train.append(document_vector(model_, doc))
    x_train = np.array(x_train) 
    x_test = []
    for doc in X_test: 
        x_test.append(document_vector(model_, doc))
    x_test = np.array(x_test) 
    # Generate freqs features: x_train/x_test
    freqs = build_freqs(X_train, y_train)
    x_train_freqs = np.zeros((len(X_train), 3))
    for i in range(len(X_train)):
        x_train_freqs[i, :]= extract_features(X_train[i], freqs)
    x_test_freqs = np.zeros((len(X_test), 3))
    for i in range(len(X_test)):
        x_test_freqs[i, :]= extract_features(X_test[i], freqs)
    if benchmark['LogisticRegression']:
        for feature_set,name in zip([[x_train,x_test],[x_train_freqs,x_test_freqs]],['GoogleEmbedding','Freqs']):
            x_train_ = feature_set[0]
            x_test_ = feature_set[1]
            clf = LogisticRegression(random_state=0, penalty ='l2' ).fit(x_train_, y_train)
            probestest = clf.predict_proba(x_test_)
            accutest = clf.score(x_test_, y_test)
            if not os.path.exists('Images/Predictions/'):  os.makedirs('Images/Predictions/')
            temp_name = 'Images/Predictions/' + str(name) + '_features_LG_Test_auc.png'
            roc(y_test, probestest[:,1], temp_name)
            print('Logistic regression {} featured testing accuracy {}'.format(name,accutest))
    if benchmark['NaiveBayes']:
        x_train_ = x_train_freqs
        x_test_ = x_test_freqs
        clfrNB = MultinomialNB(alpha = 0.1)
        clfrNB.fit(x_train_, y_train)
        preds = clfrNB.predict(x_train_)
        probestest = clfrNB.predict_proba(x_test_)
        accutest = clfrNB.score(x_test_, y_test)
        if not os.path.exists('Images/Predictions/'):  os.makedirs('Images/Predictions/')
        temp_name = 'Images/Predictions/' + str(name) + '_features_NB_Test_auc.png'
        roc(y_test, probestest[:,1], temp_name)
        print('Naive Bayes freqs featured testing accuracy {}'.format(accutest))
    if model['NN']:
        for feature_set,name in zip([[x_train,x_test],[x_train_freqs,x_test_freqs]],['GoogleEmbedding','Freqs']):
            x_train_ = feature_set[0]
            x_test_ = feature_set[1]
            x_train_, x_val, y_train_, y_val = train_test_split(x_train_, y_train, test_size=0.2, random_state=42)
            input_dim = x_train_.shape[1] 
            model = Sequential()
            model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.summary()
            history = model.fit(x_train_, np.array(y_train_),epochs=100,verbose=False,batch_size=10,validation_data=(x_val, np.array(y_val)))
            temp_name = 'Images/Predictions/' + name +'_Featured_NN_Test_TrainingVis.png'
            plot_history(history,temp_name)
            loss, accuracy = model.evaluate(x_test_, np.array(y_test), verbose=False)
            print('Neural  network {} featured testing accuracy {}'.format(name,accuracy))
            testprobes = model.predict(x_test_)
            if not os.path.exists('Images/Predictions/'):  os.makedirs('Images/Predictions/')
            temp_name = 'Images/Predictions/' + name + '_Featured_NN_Test_auc.png'
            roc(y_test, [i[0] for i in testprobes.tolist()] , temp_name)

def plot_history(history,temp_name):
    plt.style.use('ggplot')
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    if not os.path.exists('Images/Predictions/'):  os.makedirs('Images/Predictions/')
    plt.savefig(temp_name)
    plt.close()

def roc(y_test,probes,save):
    fpr, tpr, threshold = metrics.roc_curve(y_test, probes)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(save, dpi=100)
    plt.show()
    plt.close()
 
if __name__ == '__main__':
    data = pd.read_csv('Eluvio_DS_Challenge.csv')
    data['date_created'] = pd.to_datetime(data['date_created'])
    # Remove duplicate records: same title with same author and are written in the same year 
    data = data.groupby([[i.strip() for i in data['title'].tolist()],'author','over_18','category',pd.DatetimeIndex(data['date_created']).year],as_index=False
                        ).agg( {'title':'first','up_votes':sum,'down_votes':sum, 'time_created':max,'date_created':max})
    if False:
        Most_used_words(data = data)
    if False:
        Google_embedding_analysis(data = data)
    if False:
        data = data.loc[data['over_18'] == False]
        google_embedding_titles(data = data)
    if False:
        Vote_prediction = title_vote_prediction(data = data)