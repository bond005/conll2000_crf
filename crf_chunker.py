import codecs
from itertools import chain
import unittest
import nltk
import pycrfsuite

from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer


def word2shape(word):
    shape = ''
    for i in range(len(word)):
        if word[i].isalpha():
            if word[i].isupper():
                c = 'A'
            else:
                c = 'a'
        elif word[i].isdigit():
            c = 'd'
        elif word[i] == '_':
            c = '_'
        else:
            c = '*'
        if len(shape) > 0:
            if shape[-1] != c:
                shape += c
        else:
            shape = c
    return shape


def word2features(sent, i, use_shapes=True):
    word = sent[i][0].lower()
    postag = sent[i][1]
    shape = word2shape(sent[i][0])
    features = [
        'w[0]=' + word,
        'pos[0]=' + postag
    ]
    if use_shapes:
        features.append('shape[0]=' + shape)
    postag_1 = None
    if i > 0:
        word_1 = sent[i - 1][0].lower()
        shape_1 = word2shape(sent[i - 1][0])
        postag_1 = sent[i - 1][1]
        features.extend([
            'w[-1]=' + word_1,
            'pos[-1]=' + postag_1,
            'w[-1]|w[0]=' + word_1 + '|' + word,
            'pos[-1]|pos[0]=' + postag_1 + '|' + postag
        ])
        if use_shapes:
            features.extend([
                'shape[-1]=' + shape_1,
                'shape[-1]|shape[0]=' + shape_1 + '|' + shape
            ])
        if i > 1:
            word_2 = sent[i - 2][0].lower()
            shape_2 = word2shape(sent[i - 2][0])
            postag_2 = sent[i - 2][1]
            features.extend([
                'w[-2]=' + word_2,
                'pos[-2]=' + postag_2,
                'pos[-2]|pos[-1]=' + postag_2 + '|' + postag_1,
                'pos[-2]|pos[-1]|pos[0]=' + postag_2 + '|' + postag_1 + '|' + postag
            ])
            if use_shapes:
                features.append('shape[-2]=' + shape_2)
    else:
        features.append('__BOS__')
    if i < (len(sent) - 1):
        word1 = sent[i + 1][0].lower()
        shape1 = word2shape(sent[i + 1][0])
        postag1 = sent[i + 1][1]
        features.extend([
            'w[1]=' + word1,
            'pos[1]=' + postag1,
            'w[0]|w[1]=' + word + '|' + word1,
            'pos[0]|pos[1]=' + postag + '|' + postag1
        ])
        if use_shapes:
            features.extend([
                'shape[1]=' + shape1,
                'shape[0]|shape[1]=' + shape + '|' + shape1
            ])
        if postag_1 is not None:
            features.extend([
                'pos[-1]|pos[0]|pos[1]=' + postag_1 + '|' + postag + '|' + postag1
            ])
        if i < (len(sent) - 2):
            word2 = sent[i + 2][0].lower()
            shape2 = word2shape(sent[i + 2][0])
            postag2 = sent[i + 2][1]
            features.extend([
                'w[2]=' + word2,
                'pos[2]=' + postag2,
                'pos[1]|pos[2]=' + postag1 + '|' + postag2,
                'pos[0]|pos[1]|pos[2]=' + postag + '|' + postag1 + '|' + postag2
            ])
            if use_shapes:
                features.append('shape[2]=' + shape2)
    else:
        features.append('__EOS__')
    return features


def sent2tokens(sent):
    tokens = list()
    for cur_token in sent:
        if isinstance(cur_token, str):
            tokens.append(cur_token)
        else:
            if cur_token[0] == '-LRB-':
                tokens.append('(')
            elif cur_token[0] == '-RRB-':
                tokens.append(')')
            else:
                tokens.append(cur_token[0])
    return tokens


def sent2features(sent, use_shapes=True):
    prepared_sent = nltk.pos_tag(sent2tokens(sent))
    for i in range(len(prepared_sent)):
        if prepared_sent[i][0] in {',', ':', ';'}:
            prepared_sent[i] = (prepared_sent[i][0], 'PUNCT')
        elif prepared_sent[i][0] == '(':
            prepared_sent[i] = (prepared_sent[i][0], 'LRB')
        elif prepared_sent[i][0] == ')':
            prepared_sent[i] = (prepared_sent[i][0], 'RRB')
    return [word2features(prepared_sent, i, use_shapes) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, label in sent]


def load_corpus(corpus_name):
    sentences = list()
    new_sentence = list()
    with codecs.open(corpus_name, mode='r', encoding='utf-8', errors='ignore') as corpus_fp:
        line_ind = 1
        cur_line = corpus_fp.readline()
        while len(cur_line) > 0:
            prepared_line = cur_line.strip()
            if len(prepared_line) > 0:
                line_parts = prepared_line.split()
                assert len(line_parts) == 3, '{0}: line {1} is wrong!'.format(corpus_name, line_ind)
                assert (len(line_parts[0]) > 0) and (len(line_parts[1]) > 0) and (len(line_parts[2]) > 0),\
                    '{0}: line {1} is wrong!'.format(corpus_name, line_ind)
                new_sentence.append((line_parts[0], line_parts[2]))
            else:
                if len(new_sentence) > 0:
                    sentences.append(new_sentence)
                    new_sentence = list()
            cur_line = corpus_fp.readline()
            line_ind += 1
    if len(new_sentence) > 0:
        sentences.append(new_sentence)
    return sentences


def train(model_name, train_set_name, test_set_name=None, use_shapes=True, verbose=False):
    sentences_for_training = load_corpus(train_set_name)
    X_train = [sent2features(s, use_shapes) for s in sentences_for_training]
    y_train = [sent2labels(s) for s in sentences_for_training]
    if test_set_name is not None:
        sentences_for_testing = load_corpus(test_set_name)
        X_train += [sent2features(s, use_shapes) for s in sentences_for_testing]
        y_train += [sent2labels(s) for s in sentences_for_testing]
    train_params = {
        'max_iterations': 100,
        'feature.possible_states': True,
        'feature.possible_transitions': True
    }
    trainer = pycrfsuite.Trainer(algorithm='ap', params=train_params, verbose=verbose)
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)
    trainer.train(model_name)
    if verbose:
        print('')
        print(trainer.logparser.last_iteration)

		
def test(model_name, test_set_name, use_shapes=True):
    recognizer = load_recognizer(model_name)
    sentences_for_testing = load_corpus(test_set_name)
    y_pred = [recognizer.tag(sent2features(s, use_shapes)) for s in sentences_for_testing]
    y_true = [sent2labels(s) for s in sentences_for_testing]
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels=[class_indices[cls] for cls in tagset],
        target_names=tagset,
    )


def load_recognizer(model_name):
    tagger = pycrfsuite.Tagger()
    tagger.open(model_name)
    return tagger


def recognize(sentence, recognizer, use_shapes=True):
    chunks_list = list()
    source_results = recognizer.tag(sent2features(sentence, use_shapes))
    if len(source_results) == 0:
        return chunks_list
    prev_tag = source_results[0]
    new_chunk = [0]
    for cur_ind in range(len(source_results) - 1):
        cur_tag = source_results[cur_ind + 1]
        if prev_tag == 'O':
            end_of_entity = (cur_tag != 'O')
        else:
            if cur_tag == 'O':
                end_of_entity = True
            else:
                end_of_entity = cur_tag.startswith('B-') or (prev_tag[2:] != cur_tag[2:])
        if end_of_entity:
            chunks_list.append(
                (prev_tag if prev_tag == 'O' else prev_tag[2:],
                 tuple([sentence[ind] for ind in new_chunk]))
            )
            new_chunk = [cur_ind + 1]
        else:
            new_chunk.append(cur_ind + 1)
    if len(new_chunk) > 0:
        chunks_list.append(
            (prev_tag if prev_tag == 'O' else prev_tag[2:],
             tuple([sentence[ind] for ind in new_chunk]))
        )
    return chunks_list


class TestCRFChunker(unittest.TestCase):
    def test_word2shape_positive01(self):
        source_token = 'AbcdEfg'
        true_shape = 'AaAa'
        self.assertEqual(true_shape, word2shape(source_token))

    def test_word2shape_positive02(self):
        source_token = 'Abcd1Efg'
        true_shape = 'AadAa'
        self.assertEqual(true_shape, word2shape(source_token))

    def test_word2shape_positive03(self):
        source_token = 'Abcd1_Efg423'
        true_shape = 'Aad_Aad'
        self.assertEqual(true_shape, word2shape(source_token))

    def test_word2shape_positive04(self):
        source_token = 'Ab&cd1_Efg42@3'
        true_shape = 'Aa*ad_Aad*d'
        self.assertEqual(true_shape, word2shape(source_token))

    def test_sent2tokens_positive01(self):
        source_sentence = [('aaa', 'B-NP'), ('sdff45', 'I-NP'), ('gd', 'O'), ('123', 'B-VP')]
        true_list_of_tokens = ['aaa', 'sdff45', 'gd', '123']
        self.assertEqual(true_list_of_tokens, sent2tokens(source_sentence))

    def test_sent2labels_positive01(self):
        source_sentence = [('aaa', 'B-NP'), ('sdff45', 'I-NP'), ('gd', 'O'), ('123', 'B-VP')]
        true_list_of_labels = ['B-NP', 'I-NP', 'O', 'B-VP']
        self.assertEqual(true_list_of_labels, sent2labels(source_sentence))

    def test_word2features_positive01(self):
        source_sentence = [('What', 'WP'), ('would', 'MD'), ('a', 'DT'), ('Trump', 'NNP'), ('presidency', 'NN'),
                           ('mean', 'NN'), ('for', 'IN'), ('current', 'JJ'), ('international', 'JJ'),
                           ("master's", 'NN'), ('students', 'NNS'), ('on', 'IN'), ('an', 'DT'), ('F1', 'NNP'),
                           ('visa', 'NN')]
        token_index = 3
        true_features_of_token = [
            'w[0]=trump',
            'pos[0]=NNP',
            'shape[0]=Aa',
            'w[-1]=a',
            'pos[-1]=DT',
            'w[-1]|w[0]=a|trump',
            'pos[-1]|pos[0]=DT|NNP',
            'shape[-1]=a',
            'shape[-1]|shape[0]=a|Aa',
            'w[-2]=would',
            'pos[-2]=MD',
            'pos[-2]|pos[-1]=MD|DT',
            'pos[-2]|pos[-1]|pos[0]=MD|DT|NNP',
            'shape[-2]=a',
            'w[1]=presidency',
            'pos[1]=NN',
            'w[0]|w[1]=trump|presidency',
            'pos[0]|pos[1]=NNP|NN',
            'shape[1]=a',
            'shape[0]|shape[1]=Aa|a',
            'pos[-1]|pos[0]|pos[1]=DT|NNP|NN',
            'w[2]=mean',
            'pos[2]=NN',
            'pos[1]|pos[2]=NN|NN',
            'pos[0]|pos[1]|pos[2]=NNP|NN|NN',
            'shape[2]=a'
        ]
        self.assertEqual(true_features_of_token, word2features(source_sentence, token_index, use_shapes=True))

    def test_word2features_positive02(self):
        source_sentence = [('What', 'WP'), ('would', 'MD'), ('a', 'DT'), ('Trump', 'NNP'), ('presidency', 'NN'),
                           ('mean', 'NN'), ('for', 'IN'), ('current', 'JJ'), ('international', 'JJ'),
                           ("master's", 'NN'), ('students', 'NNS'), ('on', 'IN'), ('an', 'DT'), ('F1', 'NNP'),
                           ('visa', 'NN')]
        token_index = 0
        true_features_of_token = [
            'w[0]=what',
            'pos[0]=WP',
            'shape[0]=Aa',
            '__BOS__',
            'w[1]=would',
            'pos[1]=MD',
            'w[0]|w[1]=what|would',
            'pos[0]|pos[1]=WP|MD',
            'shape[1]=a',
            'shape[0]|shape[1]=Aa|a',
            'w[2]=a',
            'pos[2]=DT',
            'pos[1]|pos[2]=MD|DT',
            'pos[0]|pos[1]|pos[2]=WP|MD|DT',
            'shape[2]=a'
        ]
        self.assertEqual(true_features_of_token, word2features(source_sentence, token_index, use_shapes=True))

    def test_word2features_positive03(self):
        source_sentence = [('What', 'WP'), ('would', 'MD'), ('a', 'DT'), ('Trump', 'NNP'), ('presidency', 'NN'),
                           ('mean', 'NN'), ('for', 'IN'), ('current', 'JJ'), ('international', 'JJ'),
                           ("master's", 'NN'), ('students', 'NNS'), ('on', 'IN'), ('an', 'DT'), ('F1', 'NNP'),
                           ('visa', 'NN')]
        token_index = 14
        true_features_of_token = [
            'w[0]=visa',
            'pos[0]=NN',
            'shape[0]=a',
            'w[-1]=f1',
            'pos[-1]=NNP',
            'w[-1]|w[0]=f1|visa',
            'pos[-1]|pos[0]=NNP|NN',
            'shape[-1]=Ad',
            'shape[-1]|shape[0]=Ad|a',
            'w[-2]=an',
            'pos[-2]=DT',
            'pos[-2]|pos[-1]=DT|NNP',
            'pos[-2]|pos[-1]|pos[0]=DT|NNP|NN',
            'shape[-2]=a',
            '__EOS__'
        ]
        self.assertEqual(true_features_of_token, word2features(source_sentence, token_index, use_shapes=True))

    def test_word2features_positive04(self):
        source_sentence = [('What', 'WP'), ('would', 'MD'), ('a', 'DT'), ('Trump', 'NNP'), ('presidency', 'NN'),
                           ('mean', 'NN'), ('for', 'IN'), ('current', 'JJ'), ('international', 'JJ'),
                           ("master's", 'NN'), ('students', 'NNS'), ('on', 'IN'), ('an', 'DT'), ('F1', 'NNP'),
                           ('visa', 'NN')]
        token_index = 3
        true_features_of_token = [
            'w[0]=trump',
            'pos[0]=NNP',
            'w[-1]=a',
            'pos[-1]=DT',
            'w[-1]|w[0]=a|trump',
            'pos[-1]|pos[0]=DT|NNP',
            'w[-2]=would',
            'pos[-2]=MD',
            'pos[-2]|pos[-1]=MD|DT',
            'pos[-2]|pos[-1]|pos[0]=MD|DT|NNP',
            'w[1]=presidency',
            'pos[1]=NN',
            'w[0]|w[1]=trump|presidency',
            'pos[0]|pos[1]=NNP|NN',
            'pos[-1]|pos[0]|pos[1]=DT|NNP|NN',
            'w[2]=mean',
            'pos[2]=NN',
            'pos[1]|pos[2]=NN|NN',
            'pos[0]|pos[1]|pos[2]=NNP|NN|NN'
        ]
        self.assertEqual(true_features_of_token, word2features(source_sentence, token_index, use_shapes=False))

    def test_word2features_positive05(self):
        source_sentence = [('What', 'WP'), ('would', 'MD'), ('a', 'DT'), ('Trump', 'NNP'), ('presidency', 'NN'),
                           ('mean', 'NN'), ('for', 'IN'), ('current', 'JJ'), ('international', 'JJ'),
                           ("master's", 'NN'), ('students', 'NNS'), ('on', 'IN'), ('an', 'DT'), ('F1', 'NNP'),
                           ('visa', 'NN')]
        token_index = 0
        true_features_of_token = [
            'w[0]=what',
            'pos[0]=WP',
            '__BOS__',
            'w[1]=would',
            'pos[1]=MD',
            'w[0]|w[1]=what|would',
            'pos[0]|pos[1]=WP|MD',
            'w[2]=a',
            'pos[2]=DT',
            'pos[1]|pos[2]=MD|DT',
            'pos[0]|pos[1]|pos[2]=WP|MD|DT'
        ]
        self.assertEqual(true_features_of_token, word2features(source_sentence, token_index, use_shapes=False))

    def test_word2features_positive06(self):
        source_sentence = [('What', 'WP'), ('would', 'MD'), ('a', 'DT'), ('Trump', 'NNP'), ('presidency', 'NN'),
                           ('mean', 'NN'), ('for', 'IN'), ('current', 'JJ'), ('international', 'JJ'),
                           ("master's", 'NN'), ('students', 'NNS'), ('on', 'IN'), ('an', 'DT'), ('F1', 'NNP'),
                           ('visa', 'NN')]
        token_index = 14
        true_features_of_token = [
            'w[0]=visa',
            'pos[0]=NN',
            'w[-1]=f1',
            'pos[-1]=NNP',
            'w[-1]|w[0]=f1|visa',
            'pos[-1]|pos[0]=NNP|NN',
            'w[-2]=an',
            'pos[-2]=DT',
            'pos[-2]|pos[-1]=DT|NNP',
            'pos[-2]|pos[-1]|pos[0]=DT|NNP|NN',
            '__EOS__'
        ]
        self.assertEqual(true_features_of_token, word2features(source_sentence, token_index, use_shapes=False))


if __name__ == '__main__':
    unittest.main(verbosity=2)