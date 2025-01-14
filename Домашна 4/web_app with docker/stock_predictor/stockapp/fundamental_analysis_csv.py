# The model was created with the help of this article : https://github.com/sanidhyajadaun/Stock-Sentiment-Analysis-using-NLP
# The dataset for training the model : https://drive.google.com/file/d/1dXqHOgn8JRM88euhNlK-QywlqMs-mZOB/view

import time
from bs4 import BeautifulSoup
from selenium import webdriver
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from gensim.models import Word2Vec
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# def scrapeCompanyCodes():
#     response = requests.get("https://www.mse.mk/mk/stats/symbolhistory/STB")
#     soup = BeautifulSoup(response.text, "html.parser")
#     codesNotFormatted = soup.select("#Code option")
#     codesFormatted = []
#     for code in codesNotFormatted:
#         if not any(char.isdigit() for char in code.text):
#             codesFormatted.append(code.text)
#     return codesFormatted

def scrapeFilteredLinksForCode(code):
    from selenium import webdriver

    url = 'https://www.mse.mk/en/symbol/' + code + '/'
    webdriver = webdriver.Chrome()
    webdriver.get(url)
    time.sleep(2)

    webdriver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(5)
    soup = BeautifulSoup(webdriver.page_source,'html.parser')

    li_s = soup.select("#mCSB_2_container > ul > li")
    #uls = html.find('ul', attrs = {'id':'monsters-list'})

    filteredNewsLinks = []
    for news in li_s:
        link = news.select_one("div > a")["href"]
        headlineText = news.select_one("div > a > ul > li.flex-item-3x4 > h4").text
        if "Other price sensitive information" in headlineText:
            filteredNewsLinks.append(link)
    return filteredNewsLinks

def scrapeNewsText(links):
    from selenium import webdriver

    webdriver = webdriver.Chrome()
    news = []
    for link in links:
        url = link
        webdriver.get(url)
        time.sleep(2)
        webdriver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)
        soup = BeautifulSoup(webdriver.page_source,'html.parser')
        paragraphs = soup.select("#root > main > div > div:nth-child(4) > div > div > div > p")
        tmp = ""
        for paragraph in paragraphs:
            if len(paragraph.text) > 10:
                text = paragraph.text
                # replace "\xa0" with " "
                text = text.replace("\xa0", " ")
                tmp += text
        news.append(tmp)
    return news

def createModel():
    df = pd.read_csv(r"C:\Users\matej\Desktop\dias_project\stock_predictor\stockapp\data\Stock News Dataset.csv", encoding="ISO-8859-1")
    # dropping the unwanted columns
    df = df.drop('Date', axis=1)
    # checking the distribution of the target variable
    a = np.array(df['Label'])
    unique, counts = np.unique(a, return_counts=True)
    print(dict(zip(unique, counts)))
    # merging all the columns with the text in one column and concatenating with the target variable vertically
    df1 = df.iloc[:, 0]
    df2 = df.iloc[:, 1:26]
    df2['headlines'] = df2.apply(lambda x: ' '.join(x.astype(str)), axis=1)
    data = pd.concat([df1, df2['headlines']], axis=1)
    # removing numbers
    data['headlines'] = data['headlines'].str.replace(r'\d+(\.\d+)?', 'numbers')
    # converting into lowercase
    data['headlines'] = data['headlines'].str.lower()
    # replacing next line by a 'white space'
    data['headlines'] = data['headlines'].str.replace(r'\n', " ")
    # replacing currency sign by 'money'
    data['headlines'] = data['headlines'].str.replace(r'£|\$', 'money')
    # replacing large white space by single white space
    data['headlines'] = data['headlines'].str.replace(r'\s+', ' ')
    # replacing special characters by white space
    data['headlines'] = data['headlines'].str.replace(r"[^a-zA-Z0-9]+", " ")
    # tokenizing the documents
    list_array = []
    for i in range(len(data['headlines'])):
        temp = data['headlines'][i].split(' ')
        list_array.append(temp)
    # converting list_array into numpy array
    list_array = np.array(list_array, dtype="object")
    # performing stemming & stop word removal on the tokenized words
    stemmer = PorterStemmer()
    for i in range(len(list_array)):
        words = [stemmer.stem(word) for word in list_array[i] if word not in set(stopwords.words('english'))]
        list_array[i] = words
    # performing lemmatization
    list_headlines = []
    lemmatizer = WordNetLemmatizer()
    for i in range(len(list_array)):
        words = [lemmatizer.lemmatize(word) for word in list_array[i]]
        words = ' '.join(words)
        list_headlines.append(words)
    list_headlines = list(list_headlines)
    # bag of words
    ##cv = CountVectorizer(max_features=10)
    ##Bow = cv.fit_transform(list_headlines).toarray()
    # tf-idf
    tf = TfidfVectorizer(max_features=10)
    tfidf = tf.fit_transform(list_headlines).toarray()
    # performing lemmatization
    lemmatizer = WordNetLemmatizer()
    for i in range(len(list_array)):
        words = [lemmatizer.lemmatize(word) for word in list_array[i]]
        list_array[i] = words
    word2vec = Word2Vec(list_array, min_count=2)
    v1 = word2vec.wv['work']
    sim_words = word2vec.wv.most_similar('work')
    # splitting into training and testing data
    x = tfidf
    y = data['Label']
    x, x_test, y, y_test = train_test_split(x, y, stratify=y, test_size=0.25, random_state=42)
    # implementing multinomial naive bayes
    model = MultinomialNB()
    model.fit(x, y)
    print(model.score(x_test, y_test))
    # !!! the upper implementation gave the best results !!!
    return model

def nlpPreProcessTheNews(news):
    # creating a pandas dataframe from a list
    news = pd.DataFrame(news, columns=['News'])
    # removing numbers
    news['News'] = news['News'].str.replace(r'\d+(\.\d+)?', 'numbers')
    # converting into lowercase
    news['News'] = news['News'].str.lower()
    # replacing next line by a 'white space'
    news['News'] = news['News'].str.replace(r'\n', " ")
    # replacing currency sign by 'money'
    news['News'] = news['News'].str.replace(r'£|\$', 'money')
    # replacing large white space by single white space
    news['News'] = news['News'].str.replace(r'\s+', ' ')
    # replacing special characters by white space
    news['News'] = news['News'].str.replace(r"[^a-zA-Z0-9]+", " ")
    # tokenizing the documents
    list_array = []
    for i in range(len(news)):
        temp = news['News'][i].split(' ')
        list_array.append(temp)
    # converting list_array into numpy array
    list_array = np.array(list_array, dtype="object")
    # performing stemming & stop word removal on the tokenized words
    stemmer = PorterStemmer()
    for i in range(len(list_array)):
        words = [stemmer.stem(word) for word in list_array[i] if word not in set(stopwords.words('english'))]
        list_array[i] = words
    # performing lemmatization
    list_news = []
    lemmatizer = WordNetLemmatizer()
    for i in range(len(list_array)):
        words = [lemmatizer.lemmatize(word) for word in list_array[i]]
        words = ' '.join(words)
        list_news.append(words)
    list_news = list(list_news)
    tf = TfidfVectorizer(max_features=10)
    tfidf = tf.fit_transform(list_news).toarray()
    x = tfidf
    print(x)
    return x


# if __name__ == '__main__':
#     links = scrapeFilteredLinksForCode("ALK")
#     news = scrapeNewsText(links)
#     html_output =   """<!DOCTYPE html>
#                     <html>
#                     <head lang="mk">
#                     <meta charset="utf-8">
#                     <title>Фундаментална анализа</title>
#                     <style> body { font-family: Arial, sans-serif; background-color: #f4f4f4; margin: 0; padding: 20px; } h1 { color: #333; text-align: center; padding: 20px; } p { color: #666; text-align: justify; padding: 10px; border-bottom: 1px solid #ddd; } a { color: #007bff; text-decoration: none; } a:hover { text-decoration: underline; } </style>
#                     </head>
#                     <body>
#                     <h1>Фундаментална анализа</h1>
#                     """
#     if len(news) == 0:
#         html_output += '<p>Не се најдени вести за компанијата</p>'
#     else:
#         nlpPreProcessedNews = nlpPreProcessTheNews(news)
#         model = createModel()
#         sentimentOfNews = model.predict(nlpPreProcessedNews)
#         numOfPositiveNews = np.count_nonzero(sentimentOfNews == 1)
#         numOfNegativeNews = np.count_nonzero(sentimentOfNews == 0)
#         numOfNeutralNews = len(sentimentOfNews) - (numOfPositiveNews + numOfNegativeNews)
#         html_output += '<h2>НАПОМЕНА: Имајте на ум дека фундаменталната анализа на вестите не е секогаш точна.</h2>'
#         html_output += '<h4>Ова може да се случи поради тоа што вестите што ги објавуваат компаниите не се секогаш релевантни за предвидување на цените на акциите.</h4>'
#         html_output += '<h4>Исто така, при прегледување на објавените вести од различни компании на англиски јазик, забележано е дека некои од вестите насочуваат кон македонската верзија на вестите за да можете да ги прочитате.</h4>'
#         html_output += '<p>Вестите кои беа објавени на англиски јазик се: </p>'
#         for n in range(len(news)):
#             html_output += f'<h4>{news[n]}</h4>'
#             html_output += f'<p>Линк: <a href="{links[n]}">{links[n]}</a></p>'
#             if sentimentOfNews[n] == 1:
#                 html_output += '<p>Ова е позитивна вест.</p>'
#             elif sentimentOfNews[n] == 0:
#                 html_output += '<p>Ова е негативна вест.</p>'
#             else:
#                 html_output += '<p>Ова е неутрална вест.</p>'
#         html_output += f'<h4>Бројот на позитивни вести е: {str(numOfPositiveNews)}</h4>'
#         html_output += f'<h4>Бројот на негативни вести е: {str(numOfNegativeNews)}</h4>'
#         html_output += f'<h4>Бројот на неутрални вести е: {str(numOfNeutralNews)}</h4>'
#         if numOfPositiveNews > numOfNegativeNews:
#             html_output += '<h4>Се препорачува да се купат акции за оваа компанија.</h4>'
#         elif numOfNegativeNews > numOfPositiveNews:
#             html_output += '<h4>Не е препорачливо да се купат акции за оваа компанија, а ако доколку имате, треба да размислите за продажба.</h4>'
#         else:
#             html_output += '<h4>Бројот на позитивни и негативни вести е ист. Во овој случај не треба да се купуваат или продаваат акции.</h4>'
#     html_output += '</body></html>'
#
#     with open(r'/stock_predictor/stockapp/templates/fundamentalAnalysis.html', 'w', encoding="utf-8") as file:
#         file.write(html_output)

