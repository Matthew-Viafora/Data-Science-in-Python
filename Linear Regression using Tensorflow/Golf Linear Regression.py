import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import matplotlib.pyplot as plt
import csv
print (tf.version)

#later, make the data be imported from a csv file, but for now I am going to just put the data in a list in python

y = [68,9,4,1,2,37,40,40,21,9,50,1,5,30,10,15,20,17,40,1,6,24,40,2,31,6,4,50,23,11,55,32,5,2,12,50,23,76,15,10,50,18,50,32,50,71,69,17,50,17] #Tiger Woods placement
x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]

x = np.array(x)
y = np.array(y)


n = len(x)

print (len(y))
print(y)







#plot of training data

#plt.scatter(x,y)
#plt.ylabel("TW placement")
#plt.xlabel("Tournoment Number")
#plt.title("Training Data")
#plt.show()

#create placeholders X,Y so that I can feed training exmaples into the optizer during training process
X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(np.random.randn(), name = "W")
b = tf.Variable(np.random.randn(), name = "b")

learning_rate = 0.001
training_epochs = 1000


#hypothesis: y = Wx+b
y_prediction = tf.add(tf.multiply(X, W),  b)
#Mean Squared Error Cost Function
cost = tf.reduce_sum(tf.pow(y_prediction-Y, 2))/(2*n)
#Gradient Descent Optimzer (find minimum)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#Global Variables Initializer
init = tf.global_variables_initializer()


#Start tensorflow session
with tf.Session() as sess:

    #Initializing the variables
    sess.run(init)

    #iterating through all the epochs (1000)
    for epoch in range(training_epochs):

        #Feed each data point into the optimizer using feed Dictionary
        for (_x, _y) in zip(x,y):
            sess.run(optimizer, feed_dict = {X: _x, Y: _y})

        #Displaying the resuklt after every 50 epochs
        if (epoch+1) %50 ==0:
            #calculating the cost of every epoch
            c = sess.run(cost, feed_dict = {X: x, Y: y})
            print("Epoch", (epoch+1), ": cost =",c,"W =",sess.run(W), "b =", sess.run(b))

    #storing the necessary values to be used outside the session
    training_cost = sess.run(cost, feed_dict={X: x, Y: y})
    weight = sess.run(W)
    bias = sess.run(b)

    #calculating predictions
    predictions=weight *x+bias
    

    #new plot

    plt.plot(x,y,"ro", label = "original data")
    plt.plot(x, predictions, label = "fitted line")
    plt.title('linear regression result')
    plt.legend()
    plt.show()












