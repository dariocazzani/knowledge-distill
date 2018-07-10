# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/", validation_size=0, one_hot=True)

import tensorflow as tf
import os, tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import numpy as np

# Parameters
learning_rate = 1E-3
training_epochs = 10
batch_size = 128
input_size = 784
num_classes = 10

# Make experiments reproducible
_SEED = 42

def run_test(logits, X, Y, images, labels, dataset):
    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tot_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
    feed_dict = {X: images,
                 Y: labels,
                 is_training: False,
                 tau: 1.0}

    num_errors = len(images)-tot_correct.eval(feed_dict)
    print("******************")
    print("Accuracy on {}: {:.2f}%".format(dataset, accuracy.eval(feed_dict)*100))
    print("Num errors: {}".format(int(num_errors)))
    print("******************\n")

    return num_errors

def test_model(logits, X, Y):
    return run_test(logits, X, Y, mnist.test.images, mnist.test.labels, 'test')

def validate_model(logits, X, Y):
    return run_test(logits, X, Y, mnist.validation.images, mnist.validation.labels, 'validation')

def teacher_model(X):
    professor_X = tf.contrib.layers.xavier_initializer(uniform=True, seed=_SEED)
    reshaped = tf.reshape(X, [-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(reshaped, 32, 3, activation=tf.nn.relu, kernel_initializer=professor_X)
    conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu, kernel_initializer=professor_X)
    pool1 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(1,1))
    drop1 = tf.layers.dropout(pool1, rate=0.25, training=is_training, seed=_SEED)
    flat = tf.layers.flatten(pool1)
    dense1 = tf.layers.dense(flat, units=128, activation=tf.nn.relu,  kernel_initializer=professor_X)
    drop2 = tf.layers.dropout(dense1, rate=0.5, training=is_training, seed=_SEED)

    logits = tf.layers.dense(drop2, num_classes,  kernel_initializer=professor_X)

    return logits

def student_model(X):
    professor_X = tf.contrib.layers.xavier_initializer(uniform=True, seed=_SEED)
    dense1 = tf.layers.dense(X, units=300, activation=tf.nn.relu, kernel_initializer=professor_X)
    dense2 = tf.layers.dense(dense1, units=300, activation=tf.nn.relu, kernel_initializer=professor_X)
    logits = tf.layers.dense(dense2, num_classes, kernel_initializer=professor_X)
    return logits

if __name__ == '__main__':

    X_teacher = tf.placeholder(tf.float32, [None, input_size])
    Y_teacher = tf.placeholder(tf.float32, [None, num_classes])
    X_student = tf.placeholder(tf.float32, [None, input_size])
    Y_student = tf.placeholder(tf.float32, [None, num_classes])
    is_training = tf.placeholder(tf.bool, name='is_training')
    tau = tf.placeholder(tf.float32, name="temperature")

    """
    Teacher
    """
    logits_teacher = teacher_model(X_teacher)
    logits_teacher_with_temp = tf.div(logits_teacher, tau)

    softmax_teacher = tf.nn.softmax(logits_teacher_with_temp)
    loss_teacher = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_teacher, logits=logits_teacher_with_temp))

    optimizer_teacher = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op_teacher = optimizer_teacher.minimize(loss_teacher)

    """
    Student
    """
    logits_student = student_model(X_student)
    softmax_student = tf.nn.softmax(logits_student)
    loss_student = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_student, logits=logits_student))

    optimizer_student = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op_student = optimizer_student.minimize(loss_student)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        """
        Training Teacher cycle
        """
        print("Training teacher model...")
        epoch = 0
        while True:
            average_loss = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                feed_dict = {X_teacher: batch_x,
                             Y_teacher: batch_y,
                             is_training: True,
                             tau: 1.0}
                _, loss_value = sess.run([train_op_teacher, loss_teacher], feed_dict=feed_dict)
                # Compute average loss
                average_loss += loss_value / total_batch

            epoch+=1
            print("Epoch: {:02d} - Current loss: {:.5f}".format(epoch, average_loss))

            num_errors = test_model(logits_teacher, X_teacher, Y_teacher)
            if num_errors <= 70:
                print("Done training teacher")
                num_errors = test_model(logits_teacher, X_teacher, Y_teacher)
                break


        """
        Training Student cycle
        """
        print("\n**********\nTraining Student Model...")
        errors = []
        temperatures = np.concatenate((np.linspace(0.1, 0.8, 5), np.linspace(1, 9, 30)))
        for temperature in tqdm.tqdm(temperatures):
            print("Training student model with temperature {:.2f} ...".format(temperature))
            epoch = 0
            for epoch in range(training_epochs):
                average_loss = 0.
                total_batch = int(mnist.train.num_examples/batch_size)

                # Loop over all batches
                for i in range(total_batch):
                    batch_x, batch_y = mnist.train.next_batch(batch_size)

                    feed_dict = {X_teacher: batch_x,
                                 is_training: False,
                                 tau: temperature}

                    y_student_batch = sess.run(softmax_teacher, feed_dict=feed_dict)

                    feed_dict = {X_student: batch_x,
                                 Y_student: y_student_batch,
                                 is_training: True,
                                 tau: 1.0}

                    _, loss_value = sess.run([train_op_student, loss_student], feed_dict=feed_dict)
                    # Compute average loss
                    average_loss += loss_value / total_batch

                epoch+=1
                # print("Epoch: {:02d} - Current loss: {:.5f}".format(epoch, average_loss))

            num_errors = test_model(logits_student, X_student, Y_student)
            errors.append(num_errors)
            # print("Temperature: {} - Number of errors: {}".format(temperature, num_errors))

        plt.plot(temperatures, errors)
        plt.xticks(np.arange(0, max(temperatures)+1, 1))
        plt.xlabel('Tau')
        plt.ylabel('Errors on Test Set')
        plt.show()
