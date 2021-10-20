import re
import json
import os
import random
import pandas as pd
from utils import add_special_tokens

if not os.path.exists("./data/gpt2_1024_data/"):
    os.mkdir("./data/gpt2_1024_data/")

# Removing the emojis
def deEmojify(string):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

# Processing and cleaning the sentences
def clean_text(article, abstract):
    '''Remove unwanted characters, stopwords, and format the text to create fewer nulls word embeddings'''
    
    # Format words and remove unwanted characters
    article = re.sub(r'http\S+', '', article)
    article = re.sub(r'WWW\S+', '', article)
    article = re.sub(r'www\S+', '', article)
    # article = re.sub('[#|®™©*/@,:=+?"()_{}“”;<>\[\]\']','',article)
    article = deEmojify(article)
    article = re.sub(r'\<a href', ' ', article)
    article = re.sub(r'&amp;', '', article) 
    article = re.sub(r'<br />', ' ', article)
    article = re.sub(r'<br >', ' ', article)
    article = ' '.join(article.split())
    print(article)

    abstract = re.sub(r'http\S+', '', abstract)
    abstract = re.sub(r'WWW\S+', '', abstract)
    abstract = re.sub(r'www\S+', '', abstract)
    # abstract = re.sub('[#|®™©*/@,:=+?"()_{}“”;<>\[\]\']','',abstract)
    abstract = deEmojify(abstract)
    abstract = re.sub(r'\<a href', ' ', abstract)
    abstract = re.sub(r'&amp;', '', abstract) 
    abstract = re.sub(r'<br />', ' ', abstract)
    abstract = re.sub(r'<br >', ' ', abstract)
    abstract = ' '.join(abstract.split())
    print(abstract)

    return article, abstract

# Writing the tokens to a json file
def write_json(i,article, abstract):
	""" Saves a json file."""

	file = "./data/gpt2_1024_data/"+str(i)+".json"
	js_example = {}
	js_example['id'] = i
	js_example['article'] = article
	js_example['abstract'] = abstract
	with open(file, 'w') as f:
		json.dump(js_example, f, ensure_ascii=False)

art = []
abst = []

def main():
    tokenizer = add_special_tokens()
    print("Execution Started...")
    ids = []

    df = pd.read_excel("./data/oneliner.xlsx")

    i = 0
    for article, abstract in zip(df["Article"],df["Oneliner"]):
        article, abstract = clean_text(article, abstract)
        article, abstract = tokenizer.encode(article), tokenizer.encode(abstract)
        article = article[:900]
        art.append(len(article))
        abst.append(len(abstract))
        if len(article)>0 and len(abstract)>0 and (len(article)+len(abstract))<=1023:
            ids.append(i)
            write_json(i,article,abstract)
            i += 1
            if i%10==0:
                print(i, " files written")
        else:
            print("Error!!!!!!")
    print(f"Total {i} files written")

    valid_ids = random.sample(range(0, len(ids)), len(ids)*20//100)
    train_ids = list(set(ids) - set(valid_ids))
    print(f"Train files : {len(train_ids)}")
    print(f"Valid files : {len(valid_ids)}")
    print("Data preparation finished....")

    with open("./data/ids.json",'w') as f:
        js = {}
        js['train_ids'] = train_ids
        js['valid_ids'] = valid_ids
        #js['test_ids'] = test_ids
        json.dump(js,f)
    print(art)
    print(abst)
if __name__ == '__main__':
	main()