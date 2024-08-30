
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from scipy.special import softmax
import tweepy

# Authenticate to Twitter
consumer_key= "" #@param {type:"string"}
consumer_secret= "" #@param {type:"string"}
access_token= "" #@param {type:"string"}
access_token_secret= "" #@param {type:"string"}

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# Replace with the username of the account you want to get tweets from
username = "@InvalidUserForU" #@param {type:"string"}

# Get the user's tweets
tweets = api.user_timeline(screen_name=username, count=200, tweet_mode="extended")
tweet= tweets[0]
#tweet = "@debjit today's  weather is nice and sunny in #kolkata #weather #sunnyday 😊 https://www.google.com"

# prepocess tweet
tweet_words=[]

for word in  tweet.split(' '):
    if word.startswith('@') and len(word)>1:
        word= '@user'

    elif word.startswith('http'):
        word='http'

    tweet_words.append(word)

tweet_proc=" ".join(tweet_words)
print(tweet_proc)

#load the model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment-latest"

model= AutoModelForSequenceClassification.from_pretrained(roberta)

tokeneizer = AutoTokenizer.from_pretrained(roberta)

labels= ['Negative', 'Neutral','Positive']

#sentiment analysis

encoded_tweet=tokeneizer(tweet_proc, return_tensors='pt')

print(encoded_tweet)

output=model(encoded_tweet['input_ids'],encoded_tweet['attention_mask'])
print(output)

#get the probabilty

scores =output[0][0].detach().numpy()
scores= softmax(scores)
print(scores)

for i in range(len(scores)):
  l=labels[i]
  s=scores[i]
  print(l,s)
