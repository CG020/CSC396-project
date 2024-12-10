
import os
import string

path = f"{os.getcwd()}\\data\\train"


dictionary = dict()
for file in os.listdir(f"{os.getcwd()}\\data\\train"):
    if file.endswith(".csv"):
        print(file)
        csv = open(f"{path}\\{file}", "r", errors='ignore')
        row_number = 0
        for row in csv:
            entry = (row.strip().split(',', 3))
            text = entry[3]
            id = entry[0]
            for word in text.split():
                if len(word) > 1 and word[-1] in string.punctuation:
                    word = word[:-1]
                if len(word) > 1 and word[0] in string.punctuation:
                    word = word[1:]
                word = word.strip('"').strip('?').strip('.').strip(',').strip('!').strip(';')
                word = word.lstrip('(').rstrip(')')
                word = word.lower()  # filter with capitalization disabled
                if word.isnumeric():
                    continue
                if len(word) == 1:
                    continue
                if '@' in word:
                    continue
                if '/' in word:
                    continue
                if word == "":
                    continue
                if word[-1] == ':':
                    continue
                if word[:-1].isnumeric() and word[-1] == 's':
                    continue
                if word.replace('-', '').isnumeric():
                    continue
                if (word in dictionary):
                    dictionary[word]["count"] += 1
                    dictionary[word]["filename"].add(file)
                    dictionary[word]["id"].add(id)
                else:
                    dictionary[word] = {
                        "count": 1,
                        "filename": {file},
                        "id": {id}
                    }

max_count = dict()
max_occur = dict()
for word, data in dictionary.items():
    if (data["count"] >= 20):
        max_count[word] = data
    if (len(data["id"]) > 5):
        max_occur[word] = data

max_words = dict()
for word, data in max_count.items():
    if word in max_occur:
        max_words[word] = data["count"]

f = open(f"{path}\\all_words.txt", "w")
for word in max_words.keys():
    f.write(word + '\n')
f.close()
