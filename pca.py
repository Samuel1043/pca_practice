learning_rate = 0.01    # 0.01 this learning rate will be better! Tested
training_epochs = 10
batch_size = 256
display_step = 1000


# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)
# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])


# hidden layer settings
n_hidden_1 = 128
n_hidden_2 = 64
n_hidden_3 = 10
n_hidden_4 = 2


weights={
    'encoder_h1':tf.Variable(tf.truncated_normal([n_input,n_hidden_1])),
    'encoder_h2':tf.Variable(tf.truncated_normal([n_hidden_1,n_hidden_2])),
    'encoder_h3':tf.Variable(tf.truncated_normal([n_hidden_2,n_hidden_3])),
    'encoder_h4':tf.Variable(tf.truncated_normal([n_hidden_3,n_hidden_4])),
    'decoder_h1': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_3],)),
    'decoder_h2': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_2],)),
    'decoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1],)),
    'decoder_h4': tf.Variable(tf.truncated_normal([n_hidden_1, n_input],)),
}
biases={
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b3': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b4': tf.Variable(tf.random_normal([n_input])),

}
def encoder(x):
    layer_1=tf.sigmoid(tf.add(tf.matmul(x,weights['encoder_h1']),biases["encoder_b1"]))
    layer_2=tf.sigmoid(tf.add(tf.matmul(layer_1,weights['encoder_h2']),biases["encoder_b2"]))
    layer_3=tf.sigmoid(tf.add(tf.matmul(layer_2,weights['encoder_h3']),biases["encoder_b3"]))
    layer_4=tf.sigmoid(tf.add(tf.matmul(layer_3,weights['encoder_h4']),biases["encoder_b4"]))
    return layer_4
    
def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                biases['decoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']),
                                biases['decoder_b4']))
    return layer_4

encoder_op=encoder(X)
decoder_op=decoder(encoder_op)

y_pred=decoder_op
y_true=X


cost=tf.reduce_mean(tf.pow(y_true-y_pred,2))
optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)
    total_batch=int(mnist.train.num_examples/batch_size)
    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            _,c=sess.run([optimizer,cost],feed_dict={X:batch_xs})
            if i % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1),"cost=", "%.9f"%c)
    print("Optimization Finished!")
            
            
    encode_decode = sess.run(y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)),cmap=plt.get_cmap('gray'))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)),cmap=plt.get_cmap('gray'))
    plt.show()
    
    encoder_result = sess.run(encoder_op, feed_dict={X: mnist.test.images})
    print(type(encoder_result))
    print(type(mnist.test.labels))
    print(type(mnist.test.labels[0]))
    print(mnist.test.labels)
    index1=[]
    index2=[]
    index8=[]
    index6=[]
    index9=[]
    plt.figure(6)
    for i in range(len(mnist.test.labels)):
        if(mnist.test.labels[i]==1):
            index1.append(i)
        if(mnist.test.labels[i]==2):
            index2.append(i)
        if(mnist.test.labels[i]==8):
            index8.append(i)
        if(mnist.test.labels[i]==6):
            index6.append(i)
        if(mnist.test.labels[i]==9):
            index9.append(i)
            
    plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c=mnist.test.labels)
    plt.colorbar()
    #print(index1)
    np.reshape(testimg[i], (28, 28))
    '''
    for i in index1:
        img=np.reshape(testimg[i], (28, 28))
        label=testlabel[i]
        plt.matshow(img,cmap=plt.get_cmap('gray'))
        plt.title("" + str(i) + "th test Data " + "Label is " + str(test_label))
        c++
    '''
    plt.figure(1)
    for i in index1:
        plt.scatter(encoder_result[i,0], encoder_result[i, 1],c='grey')
    plt.figure(2)
    for i in index2:
        plt.scatter(encoder_result[i,0], encoder_result[i, 1],c="grey")
    plt.figure(3)
    for i in index8:
        plt.scatter(encoder_result[i,0], encoder_result[i, 1],c="grey")
    plt.figure(4)
    for i in index6:
        plt.scatter(encoder_result[i,0], encoder_result[i, 1],c="grey")
    plt.figure(5)
    for i in index9:
        plt.scatter(encoder_result[i,0], encoder_result[i, 1],c="grey")
        
    plt.show()
