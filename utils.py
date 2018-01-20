from __future__ import division
import math
import scipy.misc
import numpy as np
import re
import os

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def inverse_transform(images):
    return (images+1.)/2.

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def save_texts(texts, vocab, save_path):
    idss = np.argmax(texts, -1).tolist()
    with open(save_path, 'w') as f:
        for ids in idss:
            f.write(vocab.decode(ids) + '\n')

def save_texts_from_ids(texts, vocab, save_path):
    idss = texts
    with open(save_path, 'w') as f:
        for ids in idss:
            for idx in ids:
                f.write(vocab.id_to_word(idx) + ' ')
            f.write('\n')

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path, mode='RGB').astype(np.float)


# -----------------------------
class dataLayer:
    def __init__(
            self,
            image_dir='',
            text_path='',
            vocab_path='',
            dataset_name='celeba',
            image_h=64,
            image_w=64,
            batch_size=32,
            n_attributes=10,
            pre_load_images=True):

        if 'celeba' in dataset_name :
            # read texts
            f = open(text_path,'r')
            #   header
            f.readline()
            vocab = f.readline().strip().split()
            save_vocab('attr.txt', vocab)
            #   content
            cap = []
            for line in f:
                value = line.strip().split()
                c = []
                for i, attr in enumerate(value[1:]):
                    if int(attr) == 1:
                        c.append(vocab[i])
                cap.append({
                    'image_path': os.path.join(image_dir, value[0]),
                    '_text': c})
            f.close()

            # filter by length
            cap = [c for c in cap if len(c['_text']) == n_attributes] # 25003
            print(" [data] filter by length == %d, get %d texts" % (n_attributes, len(cap)))

            # load vocab
            vocab = Vocabulary(vocab_path)
            print(" [data] using vocabulary %s" % vocab_path)

            # _text_idx: encode, BOS, ..., EOS
            for c in cap:
                c['_text_idx'] = [vocab.bos_id] + vocab.encode(c['_text']) + [vocab.eos_id]
            len_mean = np.mean([len(c['_text_idx']) for c in cap])
            print(" [data] encode texts, insert BOS and EOS, length mean: %f" % len_mean)

        # pre_load_images
        if pre_load_images:
            for c in cap:
                c['image'] = self.load_image(c['image_path'], image_h, image_w)
            print(" [data] pre-load test data into memory, image_size:(%d,%d)"%(image_h, image_w))

        # split train and validation, test
        test_size = len(cap) // 10 # 10% data for testing
        test_cap = cap[-test_size:]
        val_cap = cap[(-batch_size-test_size):-test_size] # val_size = batch_size
        cap = cap[:(-batch_size-test_size)] # train_size = all - batch_size - test_size
        print(" [data] #train: %d, #test: %d" % (len(cap), len(test_cap)))

        self.cap = cap
        self.val_cap = val_cap
        self.test_cap = test_cap
        self.vocab = vocab
        self.idx = 0
        self.test_idx = 0
        self.batch_size = batch_size
        self.n_example = len(self.cap)

        # save validation images and texts
        val_images, val_texts = self.get_val_batch(batch_size)
        self.val_images_path = 'val_images.jpg'
        self.val_texts_path = 'val_texts.txt'
        self.save_images_texts(val_images, val_texts,
                self.val_images_path, self.val_texts_path)
        print(" [data] %s %s saved" % (self.val_images_path, self.val_texts_path))

    @property
    def train_size(self):
        return len(self.cap)

    @property
    def val_size(self):
        return len(self.val_cap)

    @property
    def test_size(self):
        return len(self.test_cap)

    @property
    def vocab_size(self):
        return self.vocab.size

    @property
    def val_images_save_path(self):
        return self.val_images_path

    @property
    def val_texts_save_path(self):
        return self.val_texts_path

    def save_images_texts(self, images, texts, image_path, text_path):
        save_images(np.array(images), [self.batch_size, 1], image_path)
        with open(text_path, 'w') as f:
            for cap in texts:
                for i in cap[1:]:
                    f.write(self.vocab.id_to_word(i) + ' ')
                f.write('\n')

    def load_image(self, image_path, height, width):
        img = imread(image_path)
        img = scipy.misc.imresize(img, [height, width])
        img = img/127.5 - 1
        return img

    def get_batch(self, batch_size=32):
        if self.idx + batch_size >= self.n_example:
            self.idx = 0
            np.random.shuffle(self.cap)
        images = [b['image'] for b in self.cap[self.idx:self.idx+batch_size]]
        texts = [b['_text_idx'] for b in self.cap[self.idx:self.idx+batch_size]]
        self.idx += batch_size
        return images, texts

    def get_val_batch(self, batch_size=32):
        batch = self.val_cap
        images = [b['image'] for b in batch]
        texts = [b['_text_idx'] for b in batch]
        return images, texts

    def get_test_batch(self, batch_size=32):
        if self.test_idx + batch_size >= self.test_size:
            self.test_idx = 0
        images = [b['image'] for b in self.cap[self.test_idx:(self.test_idx+batch_size)]]
        texts = [b['_text_idx'] for b in self.cap[self.test_idx:(self.test_idx+batch_size)]]
        self.test_idx += batch_size
        return images, texts

def save_vocab(filename, vocab):
    """
    vocab: list
    """
    with open(filename, 'w') as f:
        for v in vocab:
            f.write("%s\n" % v)

class Vocabulary:
    def __init__(self, vocab_filename):

        self._id_to_word = {}
        self._word_to_id = {}
        self._bos = None
        self._eos = None
        self._pad = -1
        self._unk = -1

        with open(vocab_filename, 'r') as f:
            for idx, line in enumerate(f):
                word = line.strip()
                if word == "<UNK>":
                    self._unk = idx
                if word == "<PAD>":
                    self._pad = idx
                if word == "<BOS>":
                    self._bos = idx
                if word == "<EOS>":
                    self._eos = idx
                self._id_to_word[idx] = word
                self._word_to_id[word] = idx

    @property
    def size(self):
        return len(self._id_to_word)

    @property
    def pad_id(self):
        return self._pad

    @property
    def unk_id(self):
        return self._unk

    @property
    def bos_id(self):
        return self._bos

    @property
    def eos_id(self):
        return self._eos

    def word_to_id(self, word):
        return self._word_to_id.get(word, self._unk)

    def id_to_word(self, idx):
        if idx >= self.size:
            return '<OOR>'
        return self._id_to_word.get(idx)

    def encode(self, source, do_split=True):
        if type(source) is list:
            return [self.word_to_id(word) for word in source]
        else:
            return [self.word_to_id(source)]

    def decode(self, ids):
        ids = [ids] if type(ids) is not list else ids
        return ' '.join([self.id_to_word(idx) for idx in ids])

def get_accuracy(pred_path, target_path):
    pred = open(pred_path,'r')
    pred = [line.strip().split() for line in pred]
    for p in pred:
        if "<EOS>" in p:
            p.remove("<EOS>")
    pred_dict = []
    for p in pred:
        d = {}
        for pp in p:
            d[pp] = 1
        pred_dict.append(d)

    target = open(target_path, 'r')
    target = [line.strip().split() for line in target]
    for t in target:
        if "<EOS>" in t:
            t.remove("<EOS>")

    target_dict = []
    for t in target:
        d = {}
        for tt in t:
            d[tt] = 1
        target_dict.append(d)

    accuracy = 0.0
    for p,t in zip(pred_dict, target_dict):
        value = 1.0 / len(t.keys())
        for pp in p.keys():
            if pp in t.keys():
                accuracy += value

    accuracy = accuracy / len(target_dict)
    print("%s accuracy: %.4f" % (pred_path, accuracy))
    return accuracy

def accuracy_of_random(n_choice, total_choice):
    from scipy.misc import comb
    accuracy = 0.0
    for i in range(1, n_choice+1):
        accuracy += comb(n_choice, i)*comb(total_choice-n_choice, n_choice-i)*i
    accuracy /= comb(total_choice, n_choice)
    accuracy /= n_choice
    print(accuracy)
    return accuracy

