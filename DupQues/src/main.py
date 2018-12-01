# -*- coding: utf-8 -*-
import time
import numpy as np
import sys, os, glob
import tensorflow as tf

from reader import Reader
from trainer import Trainer
from config import Configuration


class Classifier:

    def __init__(self):
        self.config = Configuration()
        self.reader = Reader(self.config)
        self.trainer = Trainer()
        self.load_data()


    def load_data(self):
        data = self.reader.load_data()
        self.train_set = data['train']
        self.val_set = data['valid']
        self.test_set = data['test']


    def train(self, restore):
        if restore == False:
            train_files = glob.glob(self.config.log_train_dir+'/*')
            for train_file in train_files:
                os.remove(train_file)

        with tf.Graph().as_default(), tf.Session() as sess:
            model = self.config.model(self.config)
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            sess.run(init)
            
            # assign word_embedding
            if self.config.word_embedding_pretrained:
                word_embedding = np.loadtxt(self.config.wordvecs_path, dtype=np.float32)
                with tf.variable_scope("Embed", reuse=True):
                    embedding1 = tf.get_variable("embedding1", 
                                                 [self.config.num_words, 
                                                  self.config.wordvec_size])
                    embedding2 = tf.get_variable("embedding2", 
                                                 [self.config.num_words, 
                                                  self.config.wordvec_size])
                    e1_assign = embedding1.assign(word_embedding)
                    e2_assign = embedding2.assign(word_embedding)
            
            sess.run([e1_assign, e2_assign])
            # assign lr
            sess.run(tf.assign(model.learning_rate, model.config.learning_rate))
            
            best_valid_acc = 0.0
            best_valid_epoch = 0

            if restore:
                print("Continue Training")
                saver.restore(sess, self.config.model_path)
            
            writer = tf.summary.FileWriter(self.config.log_graph_dir, sess.graph)
            writer.close()

            with open(self.config.log_train_acc_path, "a") as train_acc_fp,\
                 open(self.config.log_valid_acc_path, "a") as valid_acc_fp:

                for epoch in range(self.config.num_epoch):
                    start_time = time.time()

                    # whether decay lr
                    if self.config.lr_decay and epoch+1 > self.config.lr_decay_epoch:
                        lr_decay = self.config.lr_decay_rate**\
                                    max(epoch+1-self.config.lr_decay_epoch, 0.0)
                        current_lr = self.config.learning_rate*lr_decay
                        sess.run(tf.assign(model.learning_rate, current_lr))
                        
                        # stop decay if lr is below some value
                        if current_lr <= self.config.least_lr:
                            self.config.lr_decay = False

                    # train one epoch
                    print('='*40)
                    print(("Epoch %d, Learning Rate: %.6f")%(epoch+1, 
                                                             sess.run(model.learning_rate)))
                    print(("best valid acc now: %.4f") % best_valid_acc)
                    print(("best valid epoch now: %d") % best_valid_epoch)
                    loss = self.trainer.train_one_epoch(self.train_set, model, 
                                                        sess, self.config.shuffle_data)
                    print(('\ntrain loss: %.4f') % loss)

                    # evaluate on train data every 5 epoches
                    if (epoch+1)%5 == 0:
                        train_metrics = self.trainer.evaluate(self.train_set, model, sess)
                        print(('train acc: %.4f')%train_metrics["acc"])
                        train_acc_fp.write("%d: %.4f\n"%(epoch+1, train_metrics["acc"]))

                    # evaluate on valid data
                    metrics = self.trainer.evaluate(self.val_set, model, sess)
                    print(('valid acc: %.4f')%metrics["acc"])
                    valid_acc_fp.write("%d: %.4f\n"%(epoch+1, metrics["acc"]))

                    # save model if save_by_best_valid
                    if metrics["acc"] > best_valid_acc:
                        best_valid_acc = metrics["acc"]
                        best_valid_epoch = epoch+1
                        if self.config.model_save_by_best_valid:
                            saver.save(sess, self.config.model_path)

                    # save model if not save_by_best_valid
                    if not self.config.model_save_by_best_valid and \
                        (epoch+1)%self.config.model_save_period==0:
                        saver.save(sess, self.config.model_path)

                    # break if save_by_best_valid
                    if self.config.model_save_by_best_valid and \
                        epoch+1-best_valid_epoch > self.config.early_stop_epoch:
                        break

                    print("time per epoch is %.2f min"%((time.time()-start_time)/60.0))

            # save final model if not save_by_best_valid
            if not self.config.model_save_by_best_valid:
                saver.save(sess, self.config.model_path)

            print(("\nbest valid acc: %.4f")%best_valid_acc)
            metrics = self.trainer.evaluate(self.val_set, model, sess)
            print(('*'*10 + 'valid acc: %.4f')%metrics["acc"])


    def test(self):
        test_set = self.test_set
#        test_set = self.val_set
#        test_set = self.train_set
        with tf.Graph().as_default(), tf.Session() as sess:
            model = self.config.model(self.config)
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            sess.run(init)
            saver.restore(sess, self.config.model_path)

            print("testing..............")
            metrics = self.trainer.evaluate(test_set, model, sess)
            print(("accuracy: %.4f")%metrics["acc"])
            print(("precision: %.4f")%metrics["precision"])
            print(("recall: %.4f")%metrics["recall"])
            print(("F1 score: %.4f")%metrics["F1 score"])

            print(("TP: %d")%metrics["TP"])
            print(("TN: %d")%metrics["TN"])
            print(("FP: %d")%metrics["FP"])
            print(("FN: %d")%metrics["FN"])



if __name__ == "__main__":
    classifier = Classifier()

    if len(sys.argv) > 1:
        if sys.argv[1]=="continue":
            restore=True
            classifier.train(restore)
        elif sys.argv[1]=="train":
            restore = False
            classifier.train(restore)
        elif sys.argv[1]=="test":
            classifier.test()
    else:
        restore=False
        classifier.train(restore)
        classifier.test()

