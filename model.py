import tensorflow as tf

class DCGan():
    def __init__(self, sess, z_dim = 100, k=5, init_std = 0.2, eps = 1e-7,batch_size = 128,lr = 0.002):
        self.z_dim = z_dim
        self.k = k
        self.lr = lr
        self.batch_size = batch_size
        self.eps = eps
        self.sess = sess
        self.std = init_std
        self.gen_params()
        self.dis_params()
        self.model()
        self.build()

    def build(self):
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.sess.run(tf.global_variables_initializer())

    def train(self):
        for i in range(self.k):
            self.label = tf.concat([tf.ones([self.batch_size]),tf.zeros([self.batch_size])],0)
            self.loss = tf.nn.sigmoid_cross_entropy_with_logits(self.fcd, self.label)

            self.sess.run()

    def gen_params(self):
        self.fcg_w = tf.get_variable('fcg_w',[self.z_dim,1024*4*4],
                        initializer = tf.random_normal_initializer(0, self.std))

        self.fcg_b = tf.get_variable('fcg_b',[1024*4*4],initializer = tf.constant_initializer(0.0))

        self.dconv1_w = tf.get_variable('dconv1_w',[5,5,512,1024],
                        initializer = tf.random_normal_initializer(0, self.std))
        self.dconv1_b = tf.get_variable('dconv1_b',[512],initializer = tf.constant_initializer(0.0))

        self.dconv2_w = tf.get_variable('dconv2_w',[5,5,256,512],
                        initializer = tf.random_normal_initializer(0, self.std))
        self.dconv2_b = tf.get_variable('dconv2_b',[256],initializer = tf.constant_initializer(0.0))

        self.dconv3_w = tf.get_variable('dconv3_w',[5,5,128,256],
                        initializer = tf.random_normal_initializer(0, self.std))
        self.dconv3_b = tf.get_variable('dconv3_b',[128],initializer = tf.constant_initializer(0.0))

        self.dconv4_w = tf.get_variable('dconv4_w',[5,5,3,128],
                        initializer = tf.random_normal_initializer(0, self.std))
        self.dconv4_b = tf.get_variable('dconv4_b',[3],initializer = tf.constant_initializer(0.0))

    def dis_params(self):
        self.conv1_w = tf.get_variable('conv1_w',[5,5,3,64],
                        initializer = tf.random_normal_initializer(0, self.std))
        self.conv1_b = tf.get_variable('conv1_b',[64],initializer = tf.constant_initializer(0.0))

        self.conv2_w = tf.get_variable('conv2_w',[5,5,64,128],
                        initializer = tf.random_normal_initializer(0, self.std))
        self.conv2_b = tf.get_variable('conv2_b',[128],initializer = tf.constant_initializer(0.0))

        self.conv3_w = tf.get_variable('conv3_w',[5,5,128,256],
                        initializer = tf.random_normal_initializer(0, self.std))
        self.conv3_b = tf.get_variable('conv3_b',[256],initializer = tf.constant_initializer(0.0))

        self.conv4_w = tf.get_variable('conv4_w',[5,5,256,512],
                        initializer = tf.random_normal_initializer(0, self.std))
        self.conv4_b = tf.get_variable('conv4_b',[512],initializer = tf.constant_initializer(0.0))

        self.fcd_w = tf.get_variable('fcd_w',[64*64*512,1],
                        initializer = tf.random_normal_initializer(0, self.std))
        self.fcd_b = tf.get_variable('fcd_b',[1],initializer = tf.constant_initializer(0.0))


    def model(self):
        self.ginput = tf.placeholder(tf.float32, shape=(None,self.z_dim))

        with tf.variable_scope('gen_fc') as scope:
            self.fcg = tf.nn.relu(tf.matmul(self.ginput,self.fcg_w) + self.fcg_b)
            self.fcgo = tf.reshape(self.fcg,[-1,4,4,1024])
            self.fcgm, self.fcgv = tf.nn.moments(self.fcgo, [0])
            self.fcgb = tf.nn.batch_normalization(self.fcgo, self.fcgm, self.fcgv,None,None,self.eps)
            # print self.dconv1_w.shape

        with tf.variable_scope('gen_dconv1') as scope:
            self.dconv1 = tf.nn.relu(tf.nn.conv2d_transpose(self.fcgb,
                                                              self.dconv1_w,
                                                              [-1,8,8,512],
                                                              [1,1,1,1],
                                                              padding = 'VALID') + self.dconv1_b)
            self.dconv1m, self.dconv1v = tf.nn.moments(self.dconv1, [0])
            self.dconv1b = tf.nn.batch_normalization(self.dconv1, self.dconv1m, self.dconv1v ,None,None,self.eps)

        with tf.variable_scope('gen_dconv2') as scope:
            self.dconv2 = tf.nn.relu(tf.nn.conv2d_transpose(self.dconv1b,
                                                              self.dconv2_w,
                                                              [-1, 16, 16, 256],
                                                              [1,2,2,1]) + self.dconv2_b)
            self.dconv2m, self.dconv2v = tf.nn.moments(self.dconv2, [0])
            self.dconv2b = tf.nn.batch_normalization(self.dconv2, self.dconv2m, self.dconv2v ,None,None,self.eps)

        with tf.variable_scope('gen_dconv3') as scope:
            self.dconv3 = tf.nn.relu(tf.nn.conv2d_transpose(self.dconv2b,
                                                              self.dconv3_w,
                                                              [-1, 32, 32, 128],
                                                              [1,2,2,1]) + self.dconv3_b)
            self.dconv3m, self.dconv3v = tf.nn.moments(self.dconv3, [0])
            self.dconv3b = tf.nn.batch_normalization(self.dconv3, self.dconv3m, self.dconv3v ,None,None,self.eps)

        with tf.variable_scope('gen_dconv4') as scope:
            self.dconv4 = tf.nn.tanh(tf.nn.conv2d_transpose(self.dconv3b,
                                                              self.dconv4_w,
                                                              [-1, 64, 64, 3],
                                                              [1,2,2,1]) + self.dconv4_b)


        self.sampledata = tf.placeholder(tf.float32, shape = (None,64,64,3))

        self.dinput = tf.concat([self.sampledata,self.dconv4], 0)

        with tf.variable_scope('dis_conv1') as scope:
            self.conv1 = tf.nn.conv2d(self.dinput, self.conv1_w, [1,2,2,1], 'SAME') + self.conv1_b
            self.conv1r = 0.2 * self.conv1 + (1 - 0.2) * tf.nn.relu(self.conv1)
            self.conv1m, self.conv1v = tf.nn.moments(self.conv1r, [0])
            self.conv1b = tf.nn.batch_normalization(self.conv1r, self.conv1m, self.conv1v, None, None, self.eps)

        with tf.variable_scope('dis_conv2') as scope:
            self.conv2 = tf.nn.conv2d(self.conv1b, self.conv2_w, [1,2,2,1], 'SAME') + self.conv2_b
            self.conv2r = 0.2 * self.conv2 + (1 - 0.2) * tf.nn.relu(self.conv2)
            self.conv2m, self.conv2v = tf.nn.moments(self.conv2r, [0])
            self.conv2b = tf.nn.batch_normalization(self.conv2r, self.conv2m, self.conv2v, None, None, self.eps)

        with tf.variable_scope('dis_conv3') as scope:
            self.conv3 = tf.nn.conv2d(self.conv2b, self.conv3_w, [1,2,2,1], 'SAME') + self.conv3_b
            self.conv3r = 0.2 * self.conv3 + (1 - 0.2) * tf.nn.relu(self.conv3)
            self.conv3m, self.conv3v = tf.nn.moments(self.conv3r, [0])
            self.conv3b = tf.nn.batch_normalization(self.conv3r, self.conv3m, self.conv3v, None, None, self.eps)

        with tf.variable_scope('dis_conv4') as scope:
            self.conv4 = tf.nn.conv2d(self.conv3b, self.conv4_w, [1,2,2,1], 'SAME') + self.conv4_b
            self.conv4r = 0.2 * self.conv4 + (1 - 0.2) * tf.nn.relu(self.conv4)
            self.conv4m, self.conv4v = tf.nn.moments(self.conv4r, [0])
            self.conv4b = tf.nn.batch_normalization(self.conv4r, self.conv4m, self.conv4v, None, None, self.eps)

        with tf.variable_scope('dis_fc') as scope:
            self.fcd_in = tf.reshape(self.conv4b, [-1,64*64*512])
            self.fcd = tf.matmul(self.fcd_in, self.fcd_w)+ self.fcd_b

with tf.Session() as sess:
    dcgan = DCGan(sess)
    for variable in tf.global_variables():
        print variable.name
    # train_writer = tf.summary.FileWriter('./train',
                                        #   sess.graph)
