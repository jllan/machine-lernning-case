import jieba
import codecs
import numpy as np


class NativeBayes:

    def load_data(self, pos_file, neg_file):
        """载入数据"""
        neg_docs = codecs.open(neg_file, 'r', 'utf-8').readlines()
        pos_docs = codecs.open(pos_file, 'r', 'utf-8').readlines()
        return pos_docs, neg_docs

    def handle_data(self, sentence, stop_words='data/stop_words.txt'):
        """进行分词，取出停止词处理"""
        stop_words = [word.strip() for word in open(stop_words, 'r')]
        return [word.strip() for word in jieba.cut(sentence.strip()) if word and word not in stop_words]

    def create_vocal_list(self, sentence_list):
        """创建词汇表"""
        vocab_list = set([])
        for sentence in sentence_list:
            vocab_list = vocab_list | set(sentence)
        return list(vocab_list)

    def word_to_vec(self, vocab_list, sentence):
        """把一组词转换成词向量"""
        words_vec = [0]*len(vocab_list)
        for word in sentence:
            if word in vocab_list:
                words_vec[vocab_list.index(word)] = 1 # 伯努利模型，不考虑重复出现的词
        return words_vec

    def train(self, train_data, label):
        num_sentence = len(train_data)
        pos_each_word = np.zeros(len(train_data[0])) # 统计正向文本中每个词出现的次数
        neg_each_word = np.zeros(len(train_data[0])) # 统计负向文本中每个词出现的次数
        pos_words = 0.0 # 正向文本中所有词的总数
        neg_words = 0.0 # 负向文本中所有词的总数
        for i in range(num_sentence):
            if label[i] == 1:
                pos_each_word += train_data[i]
                pos_words += sum(train_data[i])
            if label[i] == 0:
                neg_each_word += train_data[i]
                neg_words += sum(train_data[i])
        pos_each_word_prob = np.log(pos_each_word/pos_words)
        neg_each_word_prob = np.log(neg_each_word/neg_words)
        print(pos_each_word_prob, neg_each_word_prob, sum(label)/num_sentence)
        return pos_each_word_prob, neg_each_word_prob, sum(label)/num_sentence

    def classify(self, test_data, pos_each_word_prob, neg_each_word_prob, pos_prob):
        p1 = sum(test_data*pos_each_word_prob) + np.log(pos_prob)
        p0 = sum(test_data*neg_each_word_prob) + np.log(1-pos_prob)
        if p1 > p0:
            return np.exp(p1),np.exp(p0)
        else:
            return np.exp(p1),np.exp(p0)

    def predict(self, test_data):
        pos_sentence_list, neg_sentence_list = self.load_data(pos_data, neg_data)
        sentences = []
        for sentence in pos_sentence_list:
            sentence = self.handle_data(sentence)
            sentences.append(sentence)
        for sentence in neg_sentence_list:
            sentence = self.handle_data(sentence)
            sentences.append(sentence)
        label = [1] * len(pos_sentence_list) + [0] * len(neg_sentence_list)
        vocab_list = self.create_vocal_list(sentences)
        train_data = []
        for sentence in sentences:
            sentence_vec = self.word_to_vec(vocab_list, sentence)
            train_data.append(sentence_vec)
        pos_each_word_prob, neg_each_word_prob, pos_prob = self.train(np.array(train_data), np.array(label))

        test_data = self.handle_data(test_data)
        test_vec = self.word_to_vec(vocab_list, test_data)
        pred = self.classify(np.array(test_vec), pos_each_word_prob, neg_each_word_prob, pos_prob)
        return pred


if __name__ == '__main__':
    pos_data = 'data/pos_6000.txt'
    neg_data = 'data/neg_6000.txt'
    # stop_words = 'data/stop_words.txt'
    native_bayes = NativeBayes()
    test_data = '价格算便宜,电风扇不好'
    print(native_bayes.predict(test_data))
