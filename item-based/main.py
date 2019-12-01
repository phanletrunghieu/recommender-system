# ref: https://github.com/codeheroku/Introduction-to-Machine-Learning/blob/master/Collaborative%20Filtering/Collaborative%20Filtering%20Dummy%20Dataset.ipynb
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

# Read data
ratings=pd.read_csv("data.csv")
ratings = ratings.drop('user', 1)
ratings.fillna(0, inplace=True)
print("read data\n", ratings)

# standardize input data
def standardize(row):
    new_row = (row - row.mean())/(row.max()-row.min())
    return new_row
ratings = ratings.apply(standardize)

# calculate simiarity
ratings = ratings.T
sparse_df = sparse.csr_matrix(ratings.values)
corrMatrix = pd.DataFrame(cosine_similarity(sparse_df),index=ratings.index,columns=ratings.index)
# the faster way
# corrMatrix = ratings.corr(method='pearson')
print(corrMatrix)

# query
similar_score = corrMatrix['abba'].sort_values(ascending=False)
print(similar_score)