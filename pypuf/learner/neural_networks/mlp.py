from numpy import reshape
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from pypuf.learner.base import Learner
from pypuf.tools import ChallengeResponseSet


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class MultiLayerPerceptron(Learner):

    def __init__(self, log_name, n, k, training_set, validation_set, transformation=None,
                 iteration_limit=1000, batch_size=1000, seed_model=None):
        self.log_name = log_name
        self.n = n
        self.k = k
        self.transformation = transformation
        self.training_set = training_set
        self.validation_set = validation_set
        self.iteration_limit = iteration_limit
        self.batch_size = min(batch_size, training_set.N)
        self.seed_model = seed_model
        self.clf = None
        self.callbacks = None
        self.history = None
        self.model = None

    def prepare(self):
        in_shape = self.n
        if self.transformation is not None:
            in_shape = self.k * self.n
            self.training_set = ChallengeResponseSet(
                challenges=self.transformation(self.training_set.challenges, self.k),
                responses=self.training_set.responses
            )
            self.validation_set = ChallengeResponseSet(
                challenges=self.transformation(self.validation_set.challenges, self.k),
                responses=self.validation_set.responses
            )
            self.training_set.challenges = reshape(self.training_set.challenges, (self.training_set.N, in_shape))
            self.validation_set.challenges = reshape(self.validation_set.challenges, (self.validation_set.N, in_shape))
        self.clf = Sequential()
        self.clf.add(Dense(self.n + 2**self.k, activation='relu', input_dim=in_shape))
        self.clf.add(Dense(self.n + 2**self.k, activation='relu'))
        self.clf.add(Dense(self.n + 2**self.k, activation='relu'))
        self.clf.add(Dense(1, activation='sigmoid'))
        self.clf.compile(
            optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        class Model:
            def __init__(self, clf, n, k, transformation):
                self.clf = clf
                self.n = n
                self.k = k
                self.transformation = transformation

            def eval(self, cs):
                if self.transformation is not None:
                    cs = reshape(self.transformation(cs, self.k), (len(cs), self.k * self.n))
                return self.clf.predict(cs)

        self.model = Model(self.clf, self.n, self.k, self.transformation)

    def learn(self):
        earlystop = EarlyStopping(monitor='loss', min_delta=0.0001, patience=10,
                                  verbose=1, mode='auto', restore_best_weights=True)
        callbacks = [earlystop]
        self.history = self.clf.fit(
            x=self.training_set.challenges,
            y=self.training_set.responses,
            batch_size=self.batch_size,
            epochs=self.iteration_limit,
            callbacks=callbacks,
            validation_data=(self.validation_set.challenges, self.validation_set.responses),
            shuffle=True
        )
        return self.model
