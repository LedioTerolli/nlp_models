import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

document = 'q 111 d rty 2342 sadf0 for are =-='
stoplist = set(stopwords.words('english'))


# tokenize document
tk = RegexpTokenizer(r'[a-zA-Z]+')
tokens = [token for token in tk.tokenize(
    document) if token not in stoplist]

# remove words with length 1
tokens = [token for token in tokens if len(token) > 1]

for token in tokens:
    print(token)