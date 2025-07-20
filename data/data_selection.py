import json
import pandas as pd 

file = open("speeches.json")
jsonString = file.read()
file.close()

jsonObject = json.loads(jsonString)
print(jsonObject[0])

# Filtering to keep Obama's speeches
obama_speeches = [d for d in jsonObject if d["president"] == "Barack Obama"]
obama_speeches_date = [d["date"] for d in obama_speeches]

# Convert in dataframe 51 speeches from 2009 to 2017
df = pd.DataFrame(obama_speeches)
df.head()
df.columns

# Saving dataframe
df = df.drop(columns=['doc_name', 'president'])
df=df.reindex(columns=['date', 'title', 'transcript'])
df.to_csv('obama_speeches.csv', index=False)
