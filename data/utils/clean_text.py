import re

from bs4 import BeautifulSoup


def clean(data):
    template = re.compile(r'https?://\S+|www\.\S+')  # Removes website links
    data = template.sub(r'', data)

    soup = BeautifulSoup(data, 'lxml')  # Removes HTML tags
    only_text = soup.get_text()
    data = only_text

    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    data = emoji_pattern.sub(r'', data)

    # clin misspells and slangs
    data = data.lower()
    data = data.replace('fk', 'fuck')
    data = data.replace('fuk', 'fuck')
    data = data.replace('f*ck', 'fuck')
    data = data.replace(r"what's", "what is ")
    data = data.replace(r"where's", "where is ")
    data = data.replace(r"'ve", " have ")
    data = data.replace(r"can't", "cannot ")
    data = data.replace(r"n't", " not ")

    data = data.replace(r"isnt", "is not ")
    data = data.replace(r"arent", "are not ")
    data = data.replace(r"dont", "do not ")
    data = data.replace(r"doesnt", "does not ")
    data = data.replace(r"cant", "can not ")
    data = data.replace(r"couldnt", "could not ")
    data = data.replace(r"shouldnt", "should not ")

    data = data.replace(r"'m", " am ")
    data = data.replace(r"'re", " are ")
    data = data.replace(r"'d", " would ")
    data = data.replace(r"'ll", " will ")
    data = data.replace(r"in'", "ing ")
    data = data.replace(r"'scuse", " excuse ")
    data = data.replace(r"'s", " ")

    # Clean some punctutations
    data = re.sub('\n', ' ', data)
    # Remove ip address
    data = re.sub(r'(([0-9]+\.){2,}[0-9]+)', ' ', data)
    data = re.sub(r'([a-zA-Z])\1{3,9}\b([0-9])\1\1\1\1', r' ', data)

    # remove time and data
    data = re.sub(r'(([0-9]+\:)[0-9]+)', ' ', data)

    # Replace repeating characters more than 3 times to length of 3
    data = re.sub(r'([*!?\'])\1\1{2,}', r'\1\1\1', data)
    # patterns with repeating characters
    data = re.sub(r'([a-zA-Z])\1{2,}\b', r'\1\1', data)
    data = re.sub(r'([a-zA-Z])\1\1{2,}\B', r'\1\1\1', data)

    # Remove repeating spaces
    data = re.sub(' +', ' ', data)

    # Ex) I didn ' t -> I didn't
    data = re.sub(" ' ", "'", data)

    return data