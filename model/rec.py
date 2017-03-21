import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from bandit import *

bandit = Bandit()

filename = "./model/data/beer_reviews.csv"
# filename = "~/Dropbox\ \(Yhat\)/yhat-box/datasets/beer_reviews/beer_reviews.csv"
df = pd.read_csv(filename)

# fix our unicode
import string
printable = set(string.printable)
df.beer_name = df.beer_name.map(lambda y: filter(lambda x: x in printable, y))
# let's limit things to the top 250
# n = 250
n = df.shape[0]

# create some summary charts:
# distribution of votes
sns.distplot(df.beer_name.value_counts())
plt.show()

bandit.metadata['>10 Votes'] = pd.value_counts(df.beer_name.value_counts() > 10)[True]
bandit.metadata.reviews = int(df.beer_name.describe()['count'])
bandit.metadata.top_beer = str(df.beer_name.describe()['top'])
bandit.metadata['50percentile'] = int(df.beer_name.value_counts().describe()['50%'])

# top_n = df.beer_name.value_counts().index[:n]
n_reviews = df.beer_name.value_counts()
quantile = .95
# we only want to recommend beers that have been reviewed by enough people
df = df[df.beer_name.isin(n_reviews[n_reviews > n_reviews.quantile(quantile)].index)]

top_beers = n_reviews[:50].index.tolist()


# df = df[df.beer_name.isin(top_n)]

print df.head()
print "melting..."
df_wide = pd.pivot_table(df, values=["review_overall"],
                         index=["beer_name", "review_profilename"],
                         aggfunc=np.mean).unstack()

# any cells that are missing data (i.e. a user didn't buy a particular product)
# we're going to set to 0
df_wide = df_wide.fillna(0)

# this is the key. we're going to use cosine_similarity from scikit-learn
# to compute the distance between all beers
print "calculating similarity"
dists = cosine_similarity(df_wide)

# stuff the distance matrix into a dataframe so it's easier to operate on
dists = pd.DataFrame(dists, columns=df_wide.index)

# give the indicies (equivalent to rownames in R) the name of the product id
dists.index = dists.columns

###############Dashboard###############
# our matrix
mx_plot = sns.heatmap(dists.iloc[:20,:20])
mx_plot.figure.savefig(bandit.output_dir + 'matrix.png')

# our ranked beers
top_reviews = df.beer_name.value_counts().reset_index()[:10]
top_reviews.columns = ['beer','reviews']
top_reviews = top_reviews.to_html(classes='table table-striped table-hover')

template = open("dashboard.html", 'r').read()
dashboard = open(bandit.output_dir + "dashboard.html", "w")

table = template.replace('{BANDIT_TABLE}', top_reviews)
img1_str = '<img src="matrix.png" style="max-height:350px;" />'
# img2_str = '<img src="dist.png" style="max-height:350px;" />'
table = table.replace('{BANDIT_PLOT_1}', img1_str)
# table = table.replace('{BANDIT_PLOT_2}', img2_str)

dashboard.write(table)
dashboard.close()

##################

def get_sims(products, n_recs=None, prob=False, unique=False):
    """
    get_top10 takes a distance matrix an a productid (assumed to be integer)
    and will calculate the 10 most similar products to product based on the
    distance matrix
    dists - a distance matrix
    product - a product id (integer)
    """
    p = dists[products].apply(lambda row: np.sum(row), axis=1)
    p = p.sort_values(ascending=False)

    p = pd.DataFrame(p).reset_index()
    p.columns = ['beer','rank']

    if n_recs == None:
        n_recs = len(p)

    # remove the inputed beers
    p = p[p.beer.isin(products) == False]

    if unique==True:
        p = p[p.beer.isin(top_beers) == False]

    if prob==False:
        p = p.drop(['rank'], axis=1)

    return p[0:n_recs]


get_sims(["Sierra Nevada Pale Ale", "60 Minute IPA"])

from yhat import Yhat, YhatModel, preprocess

class BeerRecommender(YhatModel):
    REQUIREMENTS=['numpy==1.11.3',
                  'pandas==0.19.2',
                  'scikit-learn==0.18.1']
    def execute(self, data):
        beers = data.get("beers")
        n_recs = data.get("n_recs")
        prob = data.get("prob")
        unique = data.get("unique")
        suggested_beers = get_sims(beers, n_recs, prob, unique)
        result = suggested_beers.to_dict(orient='records')
        return result

model = BeerRecommender()
model.execute({'beers':["Sierra Nevada Pale Ale"],'n_recs':10})

yh = Yhat("colin", "ce796d278f4840e30e763413d8b4baa4", "http://do-sb-dev-master.x.yhat.com/")
print yh.deploy("BeerRecommender", BeerRecommender, globals(), autodetect=False)

# print yh.predict("BeerRecommender", {"beers": ["Sierra Nevada Pale Ale",
#                  "120 Minute IPA", "Stone Ruination IPA"]})
