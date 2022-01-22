import logging
import os
import pickle

from sklearn.metrics import accuracy_score

from ml.solvers.base_solver import Solver


class SKLearnSolver(Solver):
    def __init__(self, config, args):
        super(SKLearnSolver, self).__init__(config, args)
        self.load_model()

    def load_model(self):
        self.model_filepath = os.path.join(self.result_dir, 'model.pkl')
        if os.path.exists(self.model_filepath):
            with open(self.model_filepath, 'rb') as f:
                self.model = pickle.load(f)

    def train(self):
        """
        Training the tree based networks.
        Save the model if it has better performance than the previous ones.
        """
        logging.info('Start training')
        self.model.fit(*self.train_loader.dataset.data)

        # save model
        logging.info('Saving model')
        with open(os.path.join(self.result_dir, 'model.pkl'), 'wb') as f:
            pickle.dump(self.model, f)

        self.eval()

    def eval(self):
        """
        Evaluate the model.
        """
        y_pred_less = self.model.predict(self.val_loader.dataset.inputs[0])
        y_pred_more = self.model.predict(self.val_loader.dataset.inputs[1])

        self.accuracy = (y_pred_less < y_pred_more).mean()
        print(self.accuracy)
        self.save_acc()

    def test(self):
        """
        Evaluate the model.
        """
        preds = self.model.predict(self.val_loader.dataset.inputs)

        if not self.phase == 'test':
            gt = self.val_loader.data[1]
            self.accuracy = accuracy_score(gt, preds)
            print(self.accuracy)
            self.save_acc()

        if self.config.env.save_preds:
            self.save_preds(preds)
