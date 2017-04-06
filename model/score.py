from yhat import Yhat
import string
import pandas as pd

yh = Yhat("demo-master", "3b0160e10f6d7a94a2528b11b1c9bca1", "https://sandbox.c.yhat.com/")

filename = "http://yhat-data.s3.amazonaws.com/beer_reviews.csv"
df = pd.read_csv(filename)
printable = set(string.printable)
df.beer_name = df.beer_name.map(lambda y: filter(lambda x: x in printable, y))


for row in df.beer_name.unique():
    data = { "beers": [row] }
    print yh.predict("BeerRecommender", data)
