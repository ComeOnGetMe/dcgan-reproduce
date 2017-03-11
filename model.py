import tensorflow as tf
import numpy as np
import matplotlib
import scipy.misc
# from PIL import Image

class DCGan():
    def __init__(self, sess, z_dim = 100,  k=2, init_std = 0.2, eps = 1e-7,batch_size = 128,lr = 0.0002):
        self.z_dim = z_dim
        self.k = k
        self.lr = lr
        self.batch_size = batch_size
        self.eps = eps
        self.sess = sess
        self.std = init_std
        self.image_size = (64,64,3)
        self.gen_params()
        self.dis_params()
        self.gen_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
        self.dis_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
        self.build()
        print 'model built'

    def fit(self, epochs,data, plot = True):
        print 'start fitting'
        epoch = 0
        dis_label = np.hstack([np.ones(self.batch_size),np.zeros(self.batch_size)])
        gen_label = np.ones(self.batch_size)
        while epoch < epochs:
            ind = np.random.permutation(data.shape[0])[900 * self.batch_size]
            for i in range(900):
                for j in range(self.k):
                    z = np.random.rand(self.batch_size,self.z_dim) * 2 - 1
                    sampledata = data[i * self.batch_size : (i+1)* self.batch_size]
                    _, dis_loss = self.sess.run([self.opt_dis,self.loss],
                    feed_dict ={self.ginput: z, self.sampledata : sampledata, self.label : dis_label})

                z = np.random.rand(self.batch_size,self.z_dim) * 2 - 1
                _, gen_loss = self.sess.run([self.opt_gen,self.loss],
                    feed_dict ={self.ginput: z,
                                self.sampledata :np.zeros((0,) + self.image_size) ,
                                self.label : gen_label})
                print 'batch:',i,'discriminator loss:',dis_loss,'generator loss:', gen_loss
                if plot and i % 50 == 0:
                    gimage = self.dconv4.eval(feed_dict = {self.ginput : np.random.rand(self.batch_size,self.z_dim)*2 - 1})
                    for im in range(5):
                        scipy.misc.imsave('image/'+str(epoch)+'_'+str(i)+'_'+str(im)+ '.png',(gimage[im]+1)/2)
            epoch += 1
            print 'epoch:',epoch



    def gen_params(self):
        with tf.variable_scope('generator') as scope:

            self.fcg_w = tf.get_variable('fcg_w',[self.z_dim,1024*4*4],
                            initializer = tf.random_normal_initializer(0, self.std))

            self.fcg_b = tf.get_variable('fcg_b',[1024*4*4],initializer = tf.constant_initializer(0.0))
            self.fcg_sc = tf.get_variable('fcg_sc',[1],initializer = tf.constant_initializer(1.0))
            self.fcg_sh = tf.get_variable('fcg_sh',[1],initializer = tf.constant_initializer(0.0))

            self.dconv1_w = tf.get_variable('dconv1_w',[5,5,512,1024],
                            initializer = tf.random_normal_initializer(0, self.std))
            self.dconv1_b = tf.get_variable('dconv1_b',[512],initializer = tf.constant_initializer(0.0))
            self.dconv1_sc = tf.get_variable('dconv1_sc',[1],initializer = tf.constant_initializer(1.0))
            self.dconv1_sh = tf.get_variable('dconv1_sh',[1],initializer = tf.constant_initializer(0.0))

            self.dconv2_w = tf.get_variable('dconv2_w',[5,5,256,512],
                            initializer = tf.random_normal_initializer(0, self.std))
            self.dconv2_b = tf.get_variable('dconv2_b',[256],initializer = tf.constant_initializer(0.0))
            self.dconv2_sc = tf.get_variable('dconv2_sc',[1],initializer = tf.constant_initializer(1.0))
            self.dconv2_sh = tf.get_variable('dconv2_sh',[1],initializer = tf.constant_initializer(0.0))

            self.dconv3_w = tf.get_variable('dconv3_w',[5,5,128,256],
                            initializer = tf.random_normal_initializer(0, self.std))
            self.dconv3_b = tf.get_variable('dconv3_b',[128],initializer = tf.constant_initializer(0.0))
            self.dconv3_sc = tf.get_variable('dconv3_sc',[1],initializer = tf.constant_initializer(1.0))
            self.dconv3_sh = tf.get_variable('dconv3_sh',[1],initializer = tf.constant_initializer(0.0))

            self.dconv4_w = tf.get_variable('dconv4_w',[5,5,self.image_size[-1],128],
                            initializer = tf.random_normal_initializer(0, self.std))
            self.dconv4_b = tf.get_variable('dconv4_b',[self.image_size[-1]],initializer = tf.constant_initializer(0.0))

    def dis_params(self):
        with tf.variable_scope('discriminator') as scope:
            self.conv1_w = tf.get_variable('conv1_w',[5,5,self.image_size[-1],64],
                            initializer = tf.random_normal_initializer(0, self.std))
            self.conv1_b = tf.get_variable('conv1_b',[64],initializer = tf.constant_initializer(0.0))
            self.conv1_sc = tf.get_variable('conv1_sc',[1],initializer = tf.constant_initializer(1.0))
            self.conv1_sh = tf.get_variable('conv1_sh',[1],initializer = tf.constant_initializer(0.0))

            self.conv2_w = tf.get_variable('conv2_w',[5,5,64,128],
                            initializer = tf.random_normal_initializer(0, self.std))
            self.conv2_b = tf.get_variable('conv2_b',[128],initializer = tf.constant_initializer(0.0))
            self.conv2_sc = tf.get_variable('conv2_sc',[1],initializer = tf.constant_initializer(1.0))
            self.conv2_sh = tf.get_variable('conv2_sh',[1],initializer = tf.constant_initializer(0.0))

            self.conv3_w = tf.get_variable('conv3_w',[5,5,128,256],
                            initializer = tf.random_normal_initializer(0, self.std))
            self.conv3_b = tf.get_variable('conv3_b',[256],initializer = tf.constant_initializer(0.0))
            self.conv3_sc = tf.get_variable('conv3_sc',[1],initializer = tf.constant_initializer(1.0))
            self.conv3_sh = tf.get_variable('conv3_sh',[1],initializer = tf.constant_initializer(0.0))

            self.conv4_w = tf.get_variable('conv4_w',[5,5,256,512],
                            initializer = tf.random_normal_initializer(0, self.std))
            self.conv4_b = tf.get_variable('conv4_b',[512],initializer = tf.constant_initializer(0.0))
            self.conv4_sc = tf.get_variable('conv4_sc',[1],initializer = tf.constant_initializer(1.0))
            self.conv4_sh = tf.get_variable('conv4_sh',[1],initializer = tf.constant_initializer(0.0))

            self.fcd_w = tf.get_variable('fcd_w',[4*4*512,1],
                            initializer = tf.random_normal_initializer(0, self.std))
            self.fcd_b = tf.get_variable('fcd_b',[1],initializer = tf.constant_initializer(0.0))


    def build(self):

        self.ginput = tf.placeholder(tf.float32, shape=(None,self.z_dim))

        with tf.variable_scope('gen_fc') as scope:
            self.fcg = tf.nn.relu(tf.matmul(self.ginput,self.fcg_w) + self.fcg_b)
            self.fcgo = tf.reshape(self.fcg,[-1,4,4,1024])
            self.fcgm, self.fcgv = tf.nn.moments(self.fcgo, [0])
            self.fcgb = tf.nn.batch_normalization(self.fcgo, self.fcgm, self.fcgv,self.fcg_sh,self.fcg_sc,self.eps)
            # print self.dconv1_w.shape

        with tf.variable_scope('gen_dconv1') as scope:
            self.dconv1 = tf.nn.relu(tf.nn.conv2d_transpose(self.fcgb,
                                                              self.dconv1_w,
                                                              [self.batch_size,8,8,512],
                                                              [1,1,1,1],
                                                              padding = 'VALID') + self.dconv1_b)
            self.dconv1m, self.dconv1v = tf.nn.moments(self.dconv1, [0])
            self.dconv1b = tf.nn.batch_normalization(self.dconv1, self.dconv1m, self.dconv1v,
                                                     self.dconv1_sh,self.dconv1_sc,elf.eps)

        with tf.variable_scope('gen_dconv2') as scope:
            self.dconv2 = tf.nn.relu(tf.nn.conv2d_transpose(self.dconv1b,
                                                              self.dconv2_w,
                                                              [self.batch_size, 16, 16, 256],
                                                              [1,2,2,1]) + self.dconv2_b)
            self.dconv2m, self.dconv2v = tf.nn.moments(self.dconv2, [0])
            self.dconv2b = tf.nn.batch_normalization(self.dconv2, self.dconv2m, self.dconv2v,
                                                     self.dconv2_sh,self.dconv2_sc,self.eps)

        with tf.variable_scope('gen_dconv3') as scope:
            self.dconv3 = tf.nn.relu(tf.nn.conv2d_transpose(self.dconv2b,
                                                              self.dconv3_w,
                                                              [self.batch_size, 32, 32, 128],
                                                              [1,2,2,1]) + self.dconv3_b)
            self.dconv3m, self.dconv3v = tf.nn.moments(self.dconv3, [0])
            self.dconv3b = tf.nn.batch_normalization(self.dconv3, self.dconv3m, self.dconv3v,
                                                     self.dconv3_sh,self.dconv3_sc,self.eps)

        with tf.variable_scope('gen_dconv4') as scope:
            self.dconv4 = tf.nn.tanh(tf.nn.conv2d_transpose(self.dconv3b,
                                                              self.dconv4_w,
                                                              (self.batch_size,) + self.image_size,
                                                              [1,2,2,1]) + self.dconv4_b)


        self.sampledata = tf.placeholder(tf.float32, shape = (None,)+ self.image_size)

        self.dinput = tf.concat([self.sampledata,self.dconv4], 0)

        with tf.variable_scope('dis_conv1') as scope:
            self.conv1 = tf.nn.conv2d(self.dinput, self.conv1_w, [1,2,2,1], 'SAME') + self.conv1_b
            self.conv1r = 0.2 * self.conv1 + (1 - 0.2) * tf.nn.relu(self.conv1)
            self.conv1m, self.conv1v = tf.nn.moments(self.conv1r, [0])
            self.conv1b = tf.nn.batch_normalization(self.conv1r, self.conv1m, self.conv1v,
                                                    self.conv1_sh,self.conv1_sc, self.eps)

        with tf.variable_scope('dis_conv2') as scope:
            self.conv2 = tf.nn.conv2d(self.conv1b, self.conv2_w, [1,2,2,1], 'SAME') + self.conv2_b
            self.conv2r = 0.2 * self.conv2 + (1 - 0.2) * tf.nn.relu(self.conv2)
            self.conv2m, self.conv2v = tf.nn.moments(self.conv2r, [0])
            self.conv2b = tf.nn.batch_normalization(self.conv2r, self.conv2m, self.conv2v,
                                                    self.conv2_sh,self.conv2_sc,self.eps)

        with tf.variable_scope('dis_conv3') as scope:
            self.conv3 = tf.nn.conv2d(self.conv2b, self.conv3_w, [1,2,2,1], 'SAME') + self.conv3_b
            self.conv3r = 0.2 * self.conv3 + (1 - 0.2) * tf.nn.relu(self.conv3)
            self.conv3m, self.conv3v = tf.nn.moments(self.conv3r, [0])
            self.conv3b = tf.nn.batch_normalization(self.conv3r, self.conv3m, self.conv3v,
                                                    self.conv3_sh,self.conv3_sc, self.eps)

        with tf.variable_scope('dis_conv4') as scope:
            self.conv4 = tf.nn.conv2d(self.conv3b, self.conv4_w, [1,2,2,1], 'SAME') + self.conv4_b
            self.conv4r = 0.2 * self.conv4 + (1 - 0.2) * tf.nn.relu(self.conv4)
            self.conv4m, self.conv4v = tf.nn.moments(self.conv4r, [0])
            self.conv4b = tf.nn.batch_normalization(self.conv4r, self.conv4m, self.conv4v,
                                                    self.conv4_sh,self.conv4_sc, self.eps)

        with tf.variable_scope('dis_fc') as scope:
            self.fcd_in = tf.reshape(self.conv4b, [-1,4*4*512])
            self.fcd = tf.matmul(self.fcd_in, self.fcd_w)+ self.fcd_b

        self.label = tf.placeholder(tf.float32, shape = (None))
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.fcd, labels = self.label))

        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.opt_dis = self.optimizer.minimize(self.loss, var_list = self.dis_param)
        self.opt_gen = self.optimizer.minimize(self.loss, var_list = self.gen_param)
        self.sess.run(tf.global_variables_initializer())
