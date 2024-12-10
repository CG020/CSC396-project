
import os
import string

path = f"{os.getcwd()}\\data\\train"


dictionary = dict()
for file in os.listdir(f"{os.getcwd()}\\data\\train"):
    if file.endswith(".csv"):
        print(file)
        csv = open(f"{path}\\{file}", "r")
        row_number = 0
        for row in csv:
            entry = (row.strip().split(',', 3))
            text = entry[3]
            id = entry[0]
            for word in text.split():
                word = word.strip('"').strip('?').strip('.').strip(',')
                word = word.lower()  # filter with capitalization disabled
                if word.isnumeric():
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

sorted_items = sorted(max_words.items(), key=lambda kv: (kv[1], kv[0]))

for word, count in sorted_items[::-1]:
    print(word)
