import re

from bs4 import BeautifulSoup
RE_PATTERNS = {
    ' american ': ['amerikan'],
    ' adolf ': ['adolf'],
    ' hitler ': ['hitler'],
    ' fuck': [
            '(f)(u|[^a-z0-9 ])(c|[^a-z0-9 ])(k|[^a-z0-9 ])([^ ])*',
            '(f)([^a-z]*)(u)([^a-z]*)(c)([^a-z]*)(k)',
            ' f[!@#\$%\^\&\*]*u[!@#\$%\^&\*]*k', 'f u u c',
            '(f)(c|[^a-z ])(u|[^a-z ])(k)', r'f\*',
            'feck ', ' fux ', 'f\*\*', 'f*ck', 'fk ',
            'f\-ing', 'f\.u\.', 'f###', ' fu ', 'f@ck', 'f u c k', 'f uck', 'f ck','fuk', 'wtf','fucck','f cking'],
    ' ass ': [
            '[^a-z]ass ', '[^a-z]azz ', '@\$\$', ' a\*s\*s', '[^a-z]ass[^a-z ]', 'a[@#\$%\^&\*][@#\$%\^&\*]', 'a s s'],
    ' asshole ': [' a[s|z]*wipe', 'a[s|z]*[w]*h[o|0]+[l]*e', '@\$\$hole', 'ass hole'],
    ' bitch ': [
            'b[w]*i[t]*ch', 'b!tch',
            'bi\+ch', 'b!\+ch', '(b)([^a-z]*)(i)([^a-z]*)(t)([^a-z]*)(c)([^a-z]*)(h)',
            'biatch', 'bi\*\*h', 'bytch', 'b i t c h','beetch'],
    ' bastard ': ['ba[s|z]+t[e|a]+rd'],
    ' transgender': ['transgender','trans gender'],
    ' gay ': ['gay'],
    ' cock ': [
            '[^a-z]cock', 'c0ck', '[^a-z]cok ', 'c0k', '[^a-z]cok[^aeiou]', ' cawk',
            '(c)([^a-z ])(o)([^a-z ]*)(c)([^a-z ]*)(k)', 'c o c k'
        ],
    ' dick ': [' dick[^aeiou]', 'deek', 'd i c k','diick '],
    ' suck ': ['sucker', '(s)([^a-z ]*)(u)([^a-z ]*)(c)([^a-z ]*)(k)', 'sucks', '5uck', 's u c k'],
    ' cunt ': ['cunt', 'c u n t'],
    ' bullshit ': ['bullsh\*t', 'bull\$hit','bs'],
    ' homosexual': ['homo sexual','homosex'],
    ' jerk ': ['jerk'],
    ' idiot ': ['i[d]+io[t]+', '(i)([^a-z ]*)(d)([^a-z ]*)(i)([^a-z ]*)(o)([^a-z ]*)(t)', 'idiots', 'i d i o t'],
    ' dumb ': ['(d)([^a-z ]*)(u)([^a-z ]*)(m)([^a-z ]*)(b)'],
    ' shit ': ['shitty', '(s)([^a-z ]*)(h)([^a-z ]*)(i)([^a-z ]*)(t)', 'shite', '\$hit', 's h i t'],
    ' shithole ': ['shythole','shit hole'],
    ' retard ': ['returd', 'retad', 'retard', 'wiktard', 'wikitud'],
    ' rape ': [' raped'],
    ' dumbass': ['dumb ass', 'dubass'],
    ' asshead': ['butthead', 'ass head'],
    ' sex ': ['s3x', 'sexuality'],
    ' nigger ': ['nigger', 'ni[g]+a', ' nigr ', 'negrito', 'niguh', 'n3gr', 'n i g g e r'],
    ' shut the fuck up': ['stfu'],
    ' pussy ': ['pussy[^c]', 'pusy', 'pussi[^l]', 'pusses'],
    ' faggot ': ['faggot', ' fa[g]+[s]*[^a-z ]', 'fagot', 'f a g g o t', 'faggit',
            '(f)([^a-z ]*)(a)([^a-z ]*)([g]+)([^a-z ]*)(o)([^a-z ]*)(t)', 'fau[g]+ot', 'fae[g]+ot'],
    ' motherfucker': [' motha ', ' motha f', ' mother f', 'motherucker', 'mother fucker'],
    ' whore ': ['wh\*\*\*', 'w h o r e'],
}

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

    for target, patterns in RE_PATTERNS.items():
        for pat in patterns:
            data = data.replace(pat, target)

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