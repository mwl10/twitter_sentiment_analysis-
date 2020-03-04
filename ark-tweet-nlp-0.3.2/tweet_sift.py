

file1 = open("tweets_test.txt","r+")
file2 = open("tweets_test_output.txt", "r+")

lines = file1.readlines()

for tweet in lines:
    file2.write(tweet.split("\t")[1])


