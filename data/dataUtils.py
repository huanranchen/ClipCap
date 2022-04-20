from tqdm import tqdm

def read_into_dic(file = './data/Flickr8k.lemma.token.txt'):
    '''
    :param file:  path for Flickr8k.lemma.token.txt
    :return: a dictionary. queries are image names, key are a list of captions.
    example:
1305564994_00513f9a5b.jpg#0    A man in street racer armor be examine the tire of another racer 's motorbike .
1305564994_00513f9a5b.jpg#1    Two racer drive a white bike down a road .
    your dictionary should contain:
    dic['1305564994_00513f9a5b.jpg'] =
    ['A man in street racer armor be examine the tire of another racer 's motorbike',
    'Two racer drive a white bike down a road .'
    ]
    '''
    dic = {}
    with open(file) as file_object:
        for line in file_object:
            #Divide the string into two parts
            list = line.split("\t",1)
            list_piece = list[0][:-2]
            sentence = list[1][:-1]
            if list_piece not in dic.keys():
                dic[list_piece] = []
                dic[list_piece].append(sentence)
            else:
                dic[list_piece].append(sentence)

    return dic


def split_sentence_into_words(sentence):
    '''
    :param sentence: a str like 'A man in street racer armor be examine the tire of another racer 's motorbike'
    :return:a list of words with lower case:
    ['a', 'man', 'in', 'street', ............]

    attention!!!:
    substitute any punctuations with str '<END>'!!
    if there is no punctuation at the end of the sentence, still add '<END>'

    example:
    input is 'There is a man.'
    return should be ['<START>','there','is','a','man','<END>']

    input is 'There is a man'
    return should also be ['<START>', 'there','is','a','man','<END>']
    '''
    result = []
    words_of_sentence = sentence.split(' ')
    words_of_sentence_lower = [word.lower() for word in words_of_sentence]
    if words_of_sentence_lower[-1] == '.':
        words_of_sentence_lower[-1] = '<END>'
    else:
        words_of_sentence_lower.append('<END>')
    result = words_of_sentence_lower
    result.insert(0, '<START>')

    return result


