import os
import sys

# Open files

# tweets, will be sifted to only contain the tweets themselves
tweets = open(sys.argv[1],"r+")
# <sentiment> <tab> <tagged tokens of the tweet>
output_ark = open(sys.argv[2], "r+")
# output for run tagger
tagged_tweets = open(sys.argv[3], "r+")
# have polarity in them and the tweet
tweet_lines = tweets.readlines()

# clearing tweets
tweets.truncate(0)

# getting rid of the polarity and writing back to the tweets file
for tweet in tweet_lines:
    tweet = tweet.split("\t")[1]
    tweets.write(tweet)

# make sure the tagged tweet file is empty to write to
tagged_tweets.truncate(0)

# redirect runtagger output to the tagged tweets file
runTagger_shell = "./runTagger.sh --model model.ritter_ptb_alldata_fixed.20130723 {} > {}".format(sys.argv[1], sys.argv[3])
os.system(runTagger_shell)

# get the lines from the tags
tagged_tweet_lines = tagged_tweets.readlines()

# make sure nothings in the output file first
output_ark.truncate(0)

line_index = 0
for line in tagged_tweet_lines:
    tagged_tokens = " "
    ark_tag_list = list(line.split("\t"))
    # list of words from this particular tweet
    word_list = list(ark_tag_list[0].split(" "))
    # list of tags from this particular tweet
    tag_list = list(ark_tag_list[1].split(" "))
    word_index = 0
    for word in word_list:
        # merge the word and the tag corresponding to it
        tagged_token = word_list[word_index] + "_" + tag_list[word_index] + " "
        # make the word_tag list for the entire tweet
        tagged_tokens += tagged_token
        word_index += 1
    # add the sentiment for this particular tweet
    sent_tag_token = tweet_lines[line_index].split("\t")[0] + "\t" + tagged_tokens + "\n"
    # write it to the output file
    output_ark.write(sent_tag_token)
    line_index += 1

# close the files

output_ark.close()
tagged_tweets.close()


