import DataManager, AutoEncoder, Classifier
import numpy as np
import pickle
import matplotlib.pyplot as plt


class TrainEvaluateModel:
    def __init__(self):
        self.dataManager = DataManager.DataManager()
        self.autoencoder = AutoEncoder.AutoEncoder(learning_rate=0.01)
        self.classifier = Classifier.Classifier(learning_rate=0.01)

    def train_autoencoder(self):
        batch_round = 1
        while True:
            batch_x, batch_y = self.dataManager.next_train_batch_random_select(200)
            cost = self.autoencoder.train(X=batch_x)
            if np.mod(batch_round, 10) == 0:
                print("batch_round: {0},cost={1}".format(batch_round, cost))
            if np.mod(batch_round, 100) == 0:
                self.autoencoder.save_weight()
            batch_round += 1

    def train_classifier(self):
        batch_round = 1
        while True:
            batch_x, batch_y = self.dataManager.next_train_batch_random_select(200)
            batch_x = self.autoencoder.de_noise(batch_x)
            cost = self.classifier.train(X=batch_x, Y=batch_y)
            if np.mod(batch_round, 10) == 0:
                print("batch_round: {0},cost={1}".format(batch_round, cost))
            if np.mod(batch_round, 100) == 0:
                self.classifier.save_weight()
            batch_round += 1

    def evaluate(self):
        x, y = self.dataManager.get_all_test_dataset()
        predictions = self.classifier.predict(X=x)
        print("Test Dataset distribution:[Normal, Fraud]:{0}".format(np.sum(y, axis=0)))
        evaluate_results = []
        for threshold in np.arange(0, 1.01, 0.01):
            TP, FN, FP, TN = 0, 0, 0, 0
            for i in range(len(y)):
                prediction = predictions[i]
                actual = y[i]
                if prediction[1] >= threshold and actual[1] == 1:
                    TP += 1
                elif prediction[1] >= threshold and actual[1] == 0:
                    FP += 1
                elif prediction[1] < threshold and actual[1] == 1:
                    FN += 1
                elif prediction[1] < threshold and actual[1] == 0:
                    TN += 1
            result = dict()
            result['threshold'] = threshold
            result['TP'] = TP
            result['FP'] = FP
            result['FN'] = FN
            result['TN'] = TN
            result['recall'] = TP / (TP + FN)
            result['precision'] = TP / (TP + FP)
            result['accuracy'] = (TP + TN) / (TP + FN + FP + TN)
            evaluate_results.append(result)
            print(result)
        with open('evaluate.pickle', 'wb') as output:
            pickle.dump(evaluate_results, output)
            print('evaluate result saved')

    def load_evaluation_result(self):
        with open('evaluate.pickle', 'rb') as input:
            evaluate_results = pickle.load(input)
        threshold_array = [result['threshold'] for result in evaluate_results]
        recall_array = [result['recall'] for result in evaluate_results]
        precision_array = [result['precision'] for result in evaluate_results]
        accuracy_array = [result['accuracy'] for result in evaluate_results]
        plt.plot(threshold_array, recall_array, label='recall')
        #plt.plot(threshold_array, precision_array, label='precision')
        plt.plot(threshold_array, accuracy_array, label='accuracy')
        plt.xlabel('Threshold 0~1')
        plt.ylabel('recall & precision & accuracy')
        plt.legend()
        plt.show()


model = TrainEvaluateModel()
# model.train_autoencoder()
# model.train_classifier()
# model.evaluate()
model.load_evaluation_result()

