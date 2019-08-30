import tensorflow as tf


class Classifier(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.weight_path = 'weight/Classifier.ckpt'
        self.init_model()

    def init_model(self):
        # tf Graph input
        self.X = tf.placeholder("float", [None, 29])
        self.Y = tf.placeholder("float", [None, 2])
        self.dense1 = tf.layers.dense(inputs=self.X, units=22, activation=tf.nn.leaky_relu)
        self.dense2 = tf.layers.dense(inputs=self.dense1, units=15, activation=tf.nn.leaky_relu)
        self.dense3 = tf.layers.dense(inputs=self.dense2, units=10, activation=tf.nn.leaky_relu)
        self.dense4 = tf.layers.dense(inputs=self.dense3, units=5, activation=tf.nn.leaky_relu)
        self.dense5 = tf.layers.dense(inputs=self.dense4, units=2, activation=tf.nn.leaky_relu)
        self.loss_softmax = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.dense5, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=1)
        self.train_op = self.optimizer.minimize(self.loss_softmax)

        # start session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.load_weight()

    def calc_cost(self, X, Y):
        return self.sess.run(self.loss_softmax, feed_dict={self.X: X, self.Y: Y})

    def calc_accuracy(self, X, Y):
        self.pred = tf.nn.softmax(self.dense5)
        self.correct_prediction = tf.equal(tf.argmax(self.pred, axis=1), tf.argmax(self.Y, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
        return self.sess.run(self.accuracy, feed_dict={self.X: X, self.Y: Y})

    def predict(self, X):
        self.pred = tf.nn.softmax(self.dense5)
        return self.sess.run(self.pred, feed_dict={self.X: X})

    def train(self, X, Y):
        cost, opt = self.sess.run((self.loss_softmax, self.train_op), feed_dict={self.X: X, self.Y: Y})
        return cost

    def load_weight(self):
        self.saver = tf.train.Saver()
        try:
            self.saver.restore(self.sess, self.weight_path)
            print("found saved weight.")
        except:
            print("no saved weight found.")

    def save_weight(self):
        self.saver.save(self.sess, self.weight_path)
        print("weight saved.")

