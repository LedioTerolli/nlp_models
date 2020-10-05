from gensim.corpora import Dictionary

path = 'C:\\Users\Terolli\\Desktop\\LDA model\\'

texts = [['human', 'interface', 'computer']]
dct = Dictionary(texts)  # initialize a Dictionary
dct.add_documents([["cat", "say", "meow"], ["dog"]])
print(dct.token2id)
print(dct.doc2bow(["dog", "computer", "non_existent_word"]))

file = open(path + "bigrams.txt", "a")
file.write("token")
file.close()
