import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# print(x_train[0].shape)
# imgplot = plt.imshow(x_train[0])
# plt.show()
#
# imgplot = plt.imshow(x_train[1])
# plt.show()



n_input = 784
n_hidden1 = 32
n_hidden2 = 32
n_output = 784



input_layer = tf.placeholder(tf.float32, [None, n_input])

hidden_layer_1 = {"weights": tf.Variable(tf.random_normal([n_input, n_hidden1])),
                  "biases": tf.Variable(tf.random_normal([n_hidden1]))}

hidden_layer_2 = {"weights": tf.Variable(tf.random_normal([n_hidden1, n_hidden2])),
                  "biases": tf.Variable(tf.random_normal([n_hidden2]))}

output_layer = {"weights": tf.Variable(tf.random_normal([n_hidden2, n_output])),
                  "biases": tf.Variable(tf.random_normal([n_output]))}


layer1 = tf.sigmoid(tf.matmul(input_layer, hidden_layer_1["weights"]) + hidden_layer_1["biases"])
layer2 = tf.sigmoid(tf.matmul(layer1, hidden_layer_2["weights"]) + hidden_layer_2["biases"])
output = tf.matmul(layer2, output_layer["weights"]) + output_layer["biases"]

loss = tf.reduce_mean(tf.square(output-input_layer))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)


n_epochs = 10000
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(n_epochs):
        epoch_loss = []
        for i in range(int(len(x_train)/100)):
            #i=0
            x_train_batch = x_train[i * batch_size: (i+1)*batch_size]
            _ , loss_value = sess.run([optimizer, loss], feed_dict={input_layer: x_train_batch.reshape((batch_size, n_input))})
            epoch_loss.append(loss_value)

        print("Epoch",  epoch, " Loss ", np.mean(epoch_loss))


    input_image = x_train[0]
    output_image = sess.run([output], feed_dict={input_layer: x_train[0].reshape(1, n_input)})
    output_image = output_image[0].reshape((28, 28))

    plt.imshow(input_image, cmap='gray')
    plt.show()


    plt.imshow(output_image, cmap='gray')
    plt.savefig("reconstructed_image_n_epoch"+str(n_epochs)+".png")
    plt.show()