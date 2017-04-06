from yhat import Yhat
import string
import pandas as pd

yh = Yhat("colin", "d325fc5bcb83fc197ee01edb58b4b396", "https://sandbox.c.yhat.com/")

filename = "http://yhat-data.s3.amazonaws.com/beer_reviews.csv"
df = pd.read_csv(filename)
printable = set(string.printable)
df.beer_name = df.beer_name.map(lambda y: filter(lambda x: x in printable, y))


for row in df.beer_name.unique():
    data = { "beers": [row] }
    try:
        result = yh.predict("BeerRecommender", data)['result']
        print(result[:5])
    except:
        pass
