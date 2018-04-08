import tensorflow as tf


class AutoEncoder(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.weight_path = 'weight/Autoencoder.ckpt'
        self.init_model()

    def init_model(self):
        # tf Graph input
        self.X = tf.placeholder("float", [None, 29])
        self.dense1 = tf.layers.dense(inputs=self.X, units=22, activation=tf.nn.leaky_relu)
        self.dense2 = tf.layers.dense(inputs=self.dense1, units=15, activation=tf.nn.leaky_relu)
        self.dense3 = tf.layers.dense(inputs=self.dense2, units=10, activation=tf.nn.leaky_relu)
        self.dense4 = tf.layers.dense(inputs=self.dense3, units=15, activation=tf.nn.leaky_relu)
        self.dense5 = tf.layers.dense(inputs=self.dense4, units=22, activation=tf.nn.leaky_relu)
        self.dense6 = tf.layers.dense(inputs=self.dense5, units=29, activation=tf.nn.leaky_relu)
        self.cost = tf.reduce_mean(tf.square(tf.subtract(self.dense6, self.X)))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.cost)
        # start session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.load_weight()

    def calc_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.X: X})

    def train(self, X):
        cost, opt = self.sess.run((self.cost, self.train_op), feed_dict={self.X: X})
        return cost

    def de_noise(self, X):
        return self.sess.run(self.dense6, feed_dict={self.X: X})

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
