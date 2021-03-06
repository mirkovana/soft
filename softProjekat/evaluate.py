import statistics
from difflib import SequenceMatcher

labeled_samples = dict()

with open('result - pravilni.csv', encoding='utf-8') as file:
    data = file.read()
    lines = data.split('\n')
    for index, line in enumerate(lines):
        if index == 0:
            continue
        cols = line.split(',')
        if cols and cols[0] == '':
            continue
        cols[0] = cols[0].replace('\r', '').replace("\"", "")
        cols[1] = cols[1].replace('\r', '').replace("\"", "")
        labeled_samples[str(cols[0])] = cols[1]


results = dict()

with open('result.csv', encoding='utf-8') as file:
    data = file.read()
    lines = data.split('\n')
    for index, line in enumerate(lines):
        cols = line.split(',')
        if cols and cols[0] == '':
            continue
        cols[0] = cols[0].replace('\r', '')
        cols[1] = cols[1].replace('\r', '')
        results[cols[0]] = cols[1]


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

similarities = []
for labeled_image_name in labeled_samples:
    similarity = similar(labeled_samples[labeled_image_name], results[labeled_image_name])
    similarities.append(similarity*100)

percentage = statistics.mean(similarities)
print("PROCENAT TACNOSTI RESENJA => ",percentage)
