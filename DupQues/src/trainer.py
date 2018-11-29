# -*- coding: utf-8 -*-
import sys
import random


class Trainer:

    def __init__(self):
        self.data1 = None
        self.data2 = None
        self.length1 = None
        self.length2 = None
        self.labels = None
        self.data_idxs = None


    def _feed_raw_data(self, raw_data, shuffle_flag=True):
        self.data1, self.data2, self.length1, self.length2, self.labels = raw_data
        data_idxs = list(range(len(self.labels)))
        if shuffle_flag:
            random.shuffle(data_idxs)
        self.data_idxs = data_idxs


    def _train_one_minibatch(self, step, model, sess):
        batch_size = model.batch_size
        batch_idxs = self.data_idxs[step:step + batch_size]

        input1 = [self.data1[ix] for ix in batch_idxs]
        input2 = [self.data2[ix] for ix in batch_idxs]
        length1 = [self.length1[ix] for ix in batch_idxs]
        length2 = [self.length2[ix] for ix in batch_idxs]
        labels = [self.labels[ix] for ix in batch_idxs]

        if random.random() < 0.5:
            input1, input2 = input2, input1
            length1, length2 = length2, length1
            
        loss, _ = sess.run(
                [model.loss, model.train_op],
                feed_dict = {
                    model.input1: input1,
                    model.input2: input2,
                    model.length1: length1,
                    model.length2: length2,
                    model.labels: labels,
                    model.keep_prob: 1-model.config.dropout,
                    model.training: True
                    }
                )
        return loss


    def train_one_epoch(self, raw_data, model, sess, shuffle_flag=True):
        # train one epoch
        self._feed_raw_data(raw_data, shuffle_flag=shuffle_flag)

        batch_size = model.batch_size
        output_iter = 0

        for step in range(0, len(self.labels), batch_size):
            if min(step + batch_size, len(self.labels)) < batch_size + step:
                break

            loss = self._train_one_minibatch(step, model, sess)
            percent = (step + batch_size)*100.0 / len(self.labels)

            if percent//20 > output_iter:
                output_iter = percent//20
                if output_iter < 5:
                    print('train loss: %.4f at %.2f%% of train set.\r'%(loss, percent))

            sstr = 'train loss: %.4f at %.2f%% of train set.\r'%(loss, percent)
            sys.stdout.write(sstr)
            # for linux
            sys.stdout.flush()
        return loss


    def _evaluate_one_minibatch(self, step, model, sess):
        batch_size = model.batch_size
        batch_idxs = self.data_idxs[step:step + batch_size]

        input1 = [self.data1[ix] for ix in batch_idxs]
        input2 = [self.data2[ix] for ix in batch_idxs]
        length1 = [self.length1[ix] for ix in batch_idxs]
        length2 = [self.length2[ix] for ix in batch_idxs]
        labels = [self.labels[ix] for ix in batch_idxs]

        hidden, predicts, probs = sess.run(
                [model.hidden, model.predicts, model.probabilities],
                feed_dict={
                        model.input1: input1,
                        model.input2: input2,
                        model.length1: length1,
                        model.length2: length2,
                        model.labels: labels,
                        model.keep_prob: 1.0
                        }
                )
        return (hidden, predicts, probs, labels)


    def evaluate(self, raw_data, model, sess):
        self._feed_raw_data(raw_data, shuffle_flag=False)

        batch_size = model.config.batch_size

        total_data = 0
        num_correct = 0
        predict_1 = 0
        predict_1_correct = 0
        predict_0_correct = 0
        true_1 = 0
        metrics = {}

        for step in range(0, len(self.labels), batch_size):
            if min(step+batch_size,len(self.labels)) < batch_size + step:
                break

            hidden, predicts, probs, labels = self._evaluate_one_minibatch(step, model, sess)

#            if step==1600:
#                print(hidden)
#                print(list(zip(predicts, probs)))

            for pred, label in zip(predicts, labels):
#            for prob, label in zip(probs, labels):
#                pred = 0
#                if prob > 0.37:
#                    pred = 1
                if pred == label:
                    num_correct += 1
                if pred == label == 1:
                    # TP
                    predict_1_correct += 1
                if pred == label == 0:
                    # TN
                    predict_0_correct += 1
                if pred == 1:
                    # TP + FP
                    predict_1 += 1
                if label == 1:
                    # TP + FN
                    true_1 += 1

            total_data += batch_size
            
        metrics["TP"] = predict_1_correct
        metrics["TN"] = predict_0_correct
        metrics["FP"] = predict_1 - predict_1_correct
        metrics["FN"] = true_1 - predict_1_correct
        
        metrics["acc"] = num_correct/total_data
        
        if not predict_1:
            precision = 0
        else:
            precision = predict_1_correct/predict_1
        recall = predict_1_correct/true_1
        metrics["precision"] = precision
        metrics["recall"] = recall
        if not precision and not recall:
            metrics["F1 score"] = 0
        else:
            metrics["F1 score"] = 2*precision*recall/(precision + recall)
        
        return metrics
