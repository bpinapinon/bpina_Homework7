
# %matplotlib notebook

# Import Dependencies
import tweepy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from matplotlib import style 
style.use('ggplot')
from datetime import datetime
from pprint import pprint

# Import Twitter API Keys
from BP_config import consumer_key , consumer_secret , access_token , access_token_secret

# Import Sentiment Analyzer Vader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize Vader
VaderSentimentAnalyzer = SentimentIntensityAnalyzer()

# Setup Tweepy API Authentication
Authorization = tweepy.OAuthHandler(consumer_key , consumer_secret)
Authorization.set_access_token(access_token , access_token_secret)
apiRequests = tweepy.API(Authorization , parser = tweepy.parsers.JSONParser())

####################################
# Retrive Tweet and Sentiment Data
####################################

# List of Target Users (BBC, CBS, CNN, Fox, and New York times)
TargetUsers = ["@BBCWorld", "@CBSNews", "@CNN", "@FoxNews", "@nytimes"]

# Create empty lists to store sentiments
CompoundList = []
PositiveList = []
NegativeList = []
NeutralList = []

CreateDateList = []
TweetsLagList = []
NewsOutletList = []
TweetTextList = []

# Loop through the Target Users
# x: News Outlet
for x in TargetUsers:

    # Tweet Counter
    Counter = 0

    # Retrieve the last 100 Tweets for Target User
    # y: page
    for y in range(1): 

        # Get tweets from the home feed
        UserTweets = apiRequests.user_timeline(x , page = y)

        # Loop through tweets
        # z: tweet
        for z in UserTweets: 

            # Counter Update
            Counter = Counter + 1

            # Add tweet values to associated lists
            TweetCreateDate = z["created_at"]
            TweetText = z["text"]

            CreateDateList.append(TweetCreateDate)
            TweetsLagList.append(Counter)
            NewsOutletList.append(x)
            TweetTextList.append(TweetText)

            # Run Vader Analysis on each tweet
            CompoundScore = VaderSentimentAnalyzer.polarity_scores(z["text"])["compound"]
            PositiveScore = VaderSentimentAnalyzer.polarity_scores(z["text"])["pos"]
            NegativeScore = VaderSentimentAnalyzer.polarity_scores(z["text"])["neg"]
            NeutralScore = VaderSentimentAnalyzer.polarity_scores(z["text"])["neu"]    

            # Append to the Sentiment list           
            CompoundList.append(CompoundScore)
            PositiveList.append(PositiveScore)
            NegativeList.append(NegativeScore)
            NeutralList.append(NeutralList)


# Create a data frame of the Tweet Lists created in earlier step
NewsMood_DF = pd.DataFrame({"News Outlet"     : NewsOutletList
                           ,"Tweet Date"      : CreateDateList
                           ,"Tweet Lag"       : TweetsLagList
                           ,"Tweet Text"      : TweetTextList
                           ,"Positive Score"  : PositiveList
                           ,"Negative Score"  : NegativeList                           
                           ,"Neutral Score"   : NeutralList
                           ,"Compound Score"  : CompoundList})
print(f'News Mood Data Frame:\n{"*"*25}\n{NewsMood_DF.head()}')

# Store output Data Frame as CSV
NewsMood_DF.to_csv("Output/BP_Output_NewsMoodData.csv" , index = False , header = True)

####################################
# Compound Sentiment Analysis Scatter Plot
####################################

# Partition Data by News Outlet
BBC_DF = NewsMood_DF.loc[NewsMood_DF["News Outlet"] == "@BBCWorld"]
CBS_DF = NewsMood_DF.loc[NewsMood_DF["News Outlet"] == "@CBSNews"]
CNN_DF = NewsMood_DF.loc[NewsMood_DF["News Outlet"] == "@CNN"]
FOX_DF = NewsMood_DF.loc[NewsMood_DF["News Outlet"] == "@FoxNews"]
NYT_DF = NewsMood_DF.loc[NewsMood_DF["News Outlet"] == "@nytimes"]

# Generate a Scatter Plot
BBC_TweetLag = BBC_DF["Tweet Lag"]
CBS_TweetLag = CBS_DF["Tweet Lag"]
CNN_TweetLag = CNN_DF["Tweet Lag"]
FOX_TweetLag = FOX_DF["Tweet Lag"]
NYT_TweetLag = NYT_DF["Tweet Lag"]

BBC_CompoundScore = BBC_DF["Compound Score"]
CBS_CompoundScore = CBS_DF["Compound Score"]
CNN_CompoundScore = CNN_DF["Compound Score"]
FOX_CompoundScore = FOX_DF["Compound Score"]
NYT_CompoundScore = NYT_DF["Compound Score"]

plt.scatter(BBC_TweetLag , BBC_CompoundScore , color = "red" , edgecolor = "black" , s = 100 , alpha = .9 , label = "BBC")
plt.scatter(CBS_TweetLag , CBS_CompoundScore , color = "green" , edgecolor = "black" , s = 100 , alpha = .9 , label = "CBS")
plt.scatter(CNN_TweetLag , CNN_CompoundScore , color = "blue" , edgecolor = "black" , s = 100 , alpha = .9 , label = "CNN")
plt.scatter(FOX_TweetLag , FOX_CompoundScore , color = "yellow" , edgecolor = "black" , s = 100 , alpha = .9 , label = "FOX NEWS")
plt.scatter(NYT_TweetLag , NYT_CompoundScore , color = "magenta" , edgecolor = "black" , s = 100 , alpha = .9 , label = "NYT")

# Create Plot Attributes
plt.title("Media Tweet Compound Sentiment Analysis")
plt.xlabel("Tweet Lag")
plt.ylabel("Tweet Compound Sentiment")
plt.xlim(-5 , 105)
plt.ylim(-1.5 , 1.5)
plt.legend(bbox_to_anchor = (1 , .95) , title = "News Outlet")

# Save plot as output and show
plt.savefig("Output/BP_Output_MediaTweetCompoundSentiment_ScatterPlot.png" , bbox_inches = "tight")
plt.show()


####################################
# Overall Sentiment Analysis Bar Chart
####################################

# Generate Average Compound score
AvgBBC = BBC_DF["Compound Score"].mean()
AvgCBS = CBS_DF["Compound Score"].mean()
AvgCNN = CNN_DF["Compound Score"].mean()
AvgFOX = FOX_DF["Compound Score"].mean()
AvgNYT = NYT_DF["Compound Score"].mean()

# Create attribute list for chart
OverallCompoundSentiment = [AvgBBC, AvgCBS, AvgCNN, AvgFOX, AvgNYT]
NewOutletLabels = ["BBC", "CBS", "CNN", "FOX", "NYT"]
x_Axis = np.arange(len(NewOutletLabels))
BarColor = ["red", "green", "blue", "yellow", "magenta"]

# Generate Bar Chart
plt.bar(x_Axis , OverallCompoundSentiment , color = BarColor, align = "center")
x_TicksLocation = [value for value in x_Axis]
plt.xticks(x_TicksLocation , NewOutletLabels , rotation = "vertical")

# Label Chart
plt.title("Media Tweet Overall Sentiment Analysis")
plt.xlabel("News Outet")
plt.ylabel("Average Compound Score")

#Save figure
plt.tight_layout()
plt.savefig("Output/BP_Output_MediaTweetOverallSentiment_BarChart.png")
plt.show()