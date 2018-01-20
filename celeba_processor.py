from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import image_processor
from text_processor import Vocabulary
import os
import numpy as np
import scipy.misc

class CelebaDataLayer:
    def __init__(
            self,
            image_dir='',
            text_path='',
            vocab_path='',
            image_h=64,
            image_w=64,
            n_attributes=10,
            pre_load_images=True):

        # read texts
        with open(text_path, 'r') as f:
            # header 1 is the number of texts
            f.readline()
            # header 2 is the attributes
            attribute = f.readline().strip().split()
            # content
            text = []
            for line in f:
                value = line.strip().split()
                image_path = os.path.join(image_dir, value[0])
                t = []
                for i, attr in enumerate(value[1:]):
                    if int(attr) == 1:
                        t.append(attribute[i])
                text.append({'image_path': image_path, '_text': t})
        print(" [data] %s loaded, get %d texts" % (text_path, len(text)))

        # filter by n_attributes
        if n_attributes is not None:
            text = [t for t in text if len(t['_text']) == n_attributes] # 25003
            print(" [data] filter by length == %d, get %d texts" % (n_attributes, len(text)))

        # load vocab
        vocab = Vocabulary(vocab_path)
        print(" [data] using vocabulary %s" % vocab_path)

        # _text_idx: encode, BOS, ..., EOS
        for c in text:
            c['_text_idx'] = [vocab.bos_id] + vocab.encode(c['_text']) + [vocab.eos_id]
        len_mean = np.mean([len(c['_text_idx']) for c in text])
        print(" [data] encoded texts, insert BOS and EOS, length mean: %f" % len_mean)

        # pre_load_images
        if pre_load_images:
            for c in text:
                c['image'] = self._load_image(c['image_path'], image_h, image_w)
            print(" [data] pre-loaded images into memory, image_size:(%d,%d)"%(image_h, image_w))

        # split train and validation, test
        one_tenth = len(text) // 10 # 10% data
        test_text = text[-one_tenth:]
        val_text = text[(-2 * one_tenth):-one_tenth] 
        train_text = text[:(-2 * one_tenth)]
        print(" [data] #train: %d, #validation: %d, #test: %d" % (len(train_text), len(val_text), len(test_text)))

        self.train_text = train_text
        self.val_text = val_text
        self.test_text = test_text
        self.vocab = vocab

        self.tr_idx = 0
        self.val_idx = 0
        self.test_idx = 0

        self.pre_load_images = pre_load_images

    @property
    def train_size(self):
        return len(self.train_text)

    @property
    def val_size(self):
        return len(self.val_text)

    @property
    def test_size(self):
        return len(self.test_text)

    @property
    def vocab_size(self):
        return self.vocab.size

    def _load_image(self, image_path, height=64, width=64):
        img = image_processor.read_rgb_image(image_path)
        img = image_processor.imresize(img, height, width)
        img = image_processor.transform_image(img)
        return img

    def _save_images(self, image_path, images):
        images = image_processor.inverse_transform_image(images)
        merged_image = image_processor.merged_image(images, images.shape[0], 1)
        image_processor.save_image(image_path, merged_image)

    def save_images_texts(self, images, texts, image_path, text_path):
        # save images
        images = np.array(images)
        self._save_images(image_path, images)
        # save texts
        with open(text_path, 'w') as f:
            for text in texts:
                for t in text[1:]:
                    f.write(self.vocab.id_to_word(t) + ' ')
                f.write('\n')

    # pre_load_version, if not, use images = [self._load_image(b['image_path'], 64, 64) ..]
    def get_batch(self, batch_size=32):
        if self.tr_idx + batch_size >= self.train_size:
            self.tr_idx = 0
            np.random.shuffle(self.train_text)
        images = [b['image'] for b in self.train_text[self.tr_idx:(self.tr_idx+batch_size)]]
        texts = [b['_text_idx'] for b in self.train_text[self.tr_idx:(self.tr_idx+batch_size)]]
        self.tr_idx += batch_size
        return images, texts

    def get_val_batch(self, batch_size=32):
        if self.val_idx + batch_size >= self.val_size:
            self.val_idx = 0
            np.random.shuffle(self.val_text)
        images = [b['image'] for b in self.val_text[self.val_idx:(self.val_idx+batch_size)]]
        texts = [b['_text_idx'] for b in self.val_text[self.val_idx:(self.val_idx+batch_size)]]
        return images, texts

    def get_test_batch(self, batch_size=32):
        if self.test_idx + batch_size >= self.test_size:
            self.test_idx = 0
        images = [b['image'] for b in self.test_text[self.test_idx:(self.test_idx+batch_size)]]
        texts = [b['_text_idx'] for b in self.test_text[self.test_idx:(self.test_idx+batch_size)]]
        self.test_idx += batch_size
        return images, texts

def get_accuracy(pred_path, target_path):
    """
    Args:
        pred_path: the path of the file containing predicted attributes
        target_path: the path of the file containing ground-truth attributes
    Returns:
        accuracy: the mean accuracy among all the samples
    """
    with open(pred_path, 'r') as pred:
        pred = [line.strip().split() for line in pred]
    # remove <EOS>
    for p in pred:
        if "<EOS>" in p:
            p.remove("<EOS>")
    # convert to dictionary
    pred_dict = []
    for p in pred:
        d = {}
        for pp in p:
            d[pp] = 1
        pred_dict.append(d)

    with open(target_path, 'r') as target:
        target = [line.strip().split() for line in target]
    # remove <EOS>
    for t in target:
        if "<EOS>" in t:
            t.remove("<EOS>")
    # convert to dictionary
    target_dict = []
    for t in target:
        d = {}
        for tt in t:
            d[tt] = 1
        target_dict.append(d)

    # calculate accuracy
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
    """
    Args:
        n_choice: int, number of choice
        total_choice: int, number of total choices, should be 40 for celeba
    Returns:
        accuracy: the expected mean accuracy
    """
    accuracy = 0.0
    for i in range(1, n_choice+1):
        accuracy += scipy.misc.comb(n_choice, i) * \
            scipy.misc.comb(total_choice-n_choice, n_choice-i) * \
            i
    accuracy /= scipy.misc.comb(total_choice, n_choice)
    accuracy /= n_choice
    print("expected mean accuracy of n_choice=%d, total_choice=%d: %f" % 
        (n_choice, total_choice, accuracy))
    return accuracy

