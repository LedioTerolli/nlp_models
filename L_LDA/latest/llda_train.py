from nltk.corpus import reuters
from llda import LldaModel
import pickle


# preparing reuters dataset
idlist = reuters.fileids()
idlist_train = [x for x in idlist if 'training' in x]
idlist_test = [x for x in idlist if 'test' in x]
print(len(idlist_train))
print(len(idlist_test))


# labeled_docs have the following format:
# [
#   ('reuters.words', [reuters.categories]),
#   ('reuters.words', [reuters.categories]),
#   ...
# ]
labeled_docs = [(' '.join(reuters.words(id)), reuters.categories(id))
                for id in idlist_train]


# initialize LLDA model
llda_model = LldaModel(
    labeled_documents=labeled_docs, alpha_vector=0.01)
print(llda_model)


# train llda_model
# early stop
while True:
    print("iteration %s sampling..." % (llda_model.iteration + 1))

    # training llda_model
    llda_model.training(1)

    print("after iteration: %s, perplexity: %s" %
          (llda_model.iteration, llda_model.perplexity()))
    print("delta beta: %s" % llda_model.delta_beta)
    if llda_model.is_convergent(method="beta", delta=0.01):
        break

pickle.dump(llda_model, open('llda_model.pkl', 'wb'))
