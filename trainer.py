
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
            loss, _ = sess.run(
                    [model.loss, model.train_op],
                    feed_dict = {
                        model.input1: input1,
                        model.input2: input2,
                        model.length1: length1,
                        model.length2: length2,
                        model.labels: labels,
                        }
                    )
        else:
            loss, _ = sess.run(
                    [model.loss, model.train_op],
                    feed_dict = {
                        model.input1: input2,
                        model.input2: input1,
                        model.length1: length2,
                        model.length2: length1,
                        model.labels: labels,
                        }
                    )
        return loss


    def train(self, raw_data, model, sess, shuffle_flag=True):
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
                print('train loss: %.4f at %.2f%% of training set.\r'%(loss, percent))

            sstr = 'train loss: %.4f at %.2f%% of training set.\r'%(loss, percent)
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

        predicts, probs = sess.run(
                [model.predicts, model.probabilities],
                feed_dict={
                        model.input1: input1,
                        model.input2: input2,
                        model.length1: length1,
                        model.length2: length2,
                        model.labels: labels,
                        }
                )
        return (predicts, probs, labels)


    def evaluate(self, raw_data, model, sess):
        self._feed_raw_data(raw_data, shuffle_flag=False)

        total_data = 0
        num_correct = 0
        batch_size = model.config.batch_size

        for step in range(0, len(self.labels), batch_size):
            if min(step+batch_size,len(self.labels)) < batch_size + step:
                break

            predicts, probs, labels = self._evaluate_one_minibatch(step, model, sess)

            for pred, label in zip(predicts, labels):
                if pred == label:
                    num_correct += 1

            total_data += batch_size

        acc = num_correct/total_data
        return acc
