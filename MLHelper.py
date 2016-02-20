#Machine learning includes
import numpy as np
import theano
import theano.tensor as T
import csv
import numpy
import cPickle
def _shared_dataset(data_xy):
    """ Function that loads the dataset into shared variables
    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                           dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y,
                                           dtype=theano.config.floatX))
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]


class LogisticRegression:
    def __init__(self, input, input_num,output_num):
        self.W = theano.shared(value = numpy.zeros((input_num,output_num),dtype=theano.config.floatX), name = "W",borrow=True)
        self.b = theano.shared(value= numpy.zeros((output_num,),dtype= theano.config.floatX),name = "b", borrow=True)

        self.y_pred_x = T.nnet.softmax(T.dot(input,self.W) + self.b)
        self.y_pred = T.argmax(self.y_pred_x,axis=1)
        self.params = [self.W, self.b]
        self.input = input
    def negative_log_likelihood(self,y):
        return -T.mean(T.log(self.y_pred_x)[T.arange(y.shape[0]),y])
class MLP(object):

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )
        self.hiddenTwo = HiddenLayer(
            rng = numpy.random.RandomState(121591),
            input = self.hiddenLayer.output,
            n_in = n_hidden,
            n_out = n_hidden,
            activation = T.tanh
        )
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenTwo.output,
            input_num=n_hidden,
            output_num=n_out
        )
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )

        self.params = self.hiddenLayer.params + self.hiddenTwo.params + self.logRegressionLayer.params

        self.input = input
def train(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000, batch_size=19, n_hidden=500):
    csvfile=open("fer2013.csv")

    # allocate symbolic variables for the data
    index = T.iscalar()  # index to a [mini]batch   
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=48 * 48,
        n_hidden=n_hidden,
        n_out=7
    )

    cost = (
        classifier.negative_log_likelihood(y)
        
    )

    gparams = [T.grad(cost, param) for param in classifier.params]

    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    reader = csv.DictReader(csvfile)

    x_train =numpy.zeros(shape=(28709, 48 * 48),dtype=theano.config.floatX)
    y_train =numpy.zeros(shape=(28709,),dtype=theano.config.floatX)
   
    cur_index=0
    for row in reader:
        x_train[cur_index] =map(float,row["pixels"].split(" "))
        y_train[cur_index] = row["emotion"]    
    train = [x_train, y_train]
    train_a =_shared_dataset(train)

    train_model = theano.function(
            inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_a[0][index * batch_size: (index + 1) * batch_size],
            y: train_a[1][index * batch_size: (index + 1) * batch_size]
        }
    )

    f = file("EmotionalLearning.save","wb")
    for row in range(0,(28709/batch_size)):
	train_model(row)
        print((28709 / batch_size) - row, " remaining operations")

    cPickle.dump([param.get_value() for param in classifier.params],f, protocol=cPickle.HIGHEST_PROTOCOL)		
if __name__ == "__main__":
	train()
	
	

