# -*- coding: utf-8 -*-
import nltk
nltk.download('punkt')
import tflearn, tensorflow, random, json, pickle
import numpy as np
from nltk.stem.lancaster import LancasterStemmer

# !!!ВАЖНО!!!
# я подробно пишу там, что важно для других предприятий
# а, то-что вам не нужно, я пишу кратко

# создаём класс с нейросетью
class SwearInsultAI():
    def __init__(self, message, justInsultOtherPerson = False, type = "everything"):
        self.stemmer = LancasterStemmer()
        with open("data.json", encoding="utf8") as file: # получаем значение данных на которых обучалась нейросеть
            self.data = json.load(file)
        with open("data.pickle", "rb") as f: # получаем результат обучения нейросети
            self.words, self.labels, self.training, self.output = pickle.load(f)
        self.message = message # само сообщение
        self.type = type # тип сообщений которые мы считаем проблемными. Варианты (insult, swear, или что-то что не равное двум предыдущим)
        self.justInsultOtherPerson = justInsultOtherPerson # если True, значит мы считаем сообщение проблемных только в случае если человек оскорбляет других людей. А если False, тогда нельзя оскорблять себя и других

    def load_nn(self):
        # создаём нейросеть, добавляем нейроны, выбираем функцию активации softmax(так-как нейронка для классификации)
        net = tflearn.input_data(shape=[None, len(self.training[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(self.output[0]), activation="softmax")
        net = tflearn.regression(net)
        model = tflearn.DNN(net)
        model.load("model.tflearn") # загружаем данные в нашу нейронку
        return model # возвращаем подгруженную нейросеть

    def bag_of_words(self, s, words):
        # мы выбираем значения из data.json(данных ДЛЯ обучения)
        bag = [0 for _ in range(len(words))]
        s_words = nltk.word_tokenize(s)
        s_words = [self.stemmer.stem(word.lower()) for word in s_words]
        for se in s_words:
            for i, w in enumerate(words):
                if w == se:
                    bag[i] = 1
        return np.array(bag)

    def check_message(self):
        # это главная функция класса
        symbols = ',.-=[]{}\;"\|/+*1234567890?`~!@#№$%^:&' # это символы которые мы будем удалать из сообщений, это нужно для того чтобы было легче понимать сообщение нейросети, и в последствии классифицировать сообщение как ("С матом, с оскорблением, и с тем, и стем, или как нормальное сообщение")
        self.message = self.message.lower()
        for symbol in symbols:
            self.message = self.message.replace(symbol, " ") # удаляем символы из сообщения
        results = self.load_nn().predict([self.bag_of_words(self.message, self.words)]) # тут мы уже подключем нейросеть, и она возвратит список в списке с вероятностями на сколько он похож из списка возможных(insult(оскорбление), swear(мат), empty(пустое, т.е. сообщение нормальное))
        if self.type == "insult": # если тот кто использует нейросеть указал что мы ищем только оскорбления в сообщении, тогда вероятность похождения на сообщение с матами будет равное нулю
            results[0][2] = 0.0
        elif self.type == "swearing": # тоже самое, только, мы ищем в сообщении маты, а не оскорбление, и делаем вероятность похождение на сообщение с оскорблениями будет равное нулю
            results[0][1] = 0.0
        results_index = np.argmax(results) # получаем индекс того на что он похож
        tag = self.labels[results_index] # ставим тег(insult(оскорбление), swear(мат), empty(пустое, т.е. сообщение нормальное)
        for tg in self.data["data"]:
            if tg['tag'] == tag: # находим в словаре с данными для обучения - тэг
                self.message = " " + self.message # добавляем в конец сообщения пробел, чтобы он мог понять что если человек сам себя оскорбляет(+ если justInsultOtherPerson равное True)
                if (tag == "insult" and not (" я " in self.message) and self.justInsultOtherPerson) or tag == "swearing" or (tag == "insult" and not self.justInsultOtherPerson): # просто проверяю все возможные варианты, также основываясь на данных данные пользоватилем
                    return True # сообщение - проблемное
                else:
                    return False # сообщение - нормальное
# Пример использования:
# ai = SwearInsultAI("блять!", False)
# print(ai.check_message())