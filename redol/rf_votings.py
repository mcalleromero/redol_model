import numpy as np

from sklearn.ensemble import RandomForestClassifier


class RandomForestVotings:
    def __init__(self, n_estimators=100):
        self.model = RandomForestClassifier(n_estimators)

    def fit(self, X_train, y_train):
        self.fit_model = self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def get_margins(self, X_test, y_test):
        """This function gets the margins for each instance

        Args:
            X_test : Test features
            y_test : Test labels

        Returns:
            margins : margins for each instance of the test dataset
            X_predictions : predictions made by the margins
        """
        if len(np.unique(y_test)) > 2:
            raise ValueError('Classification problem must be binary.')

        margins = np.zeros_like(y_test)

        probs = self.model.predict_proba(X_test)
        margins[y_test == 0] = probs[y_test == 0, 0] - probs[y_test == 0, 1]
        margins[y_test == 1] = probs[y_test == 1, 1] - probs[y_test == 1, 0]
        margins /= np.sum(probs, axis=1)

        return margins, np.where(margins > 0, y_test, 1 - y_test)

        # margins = []
        # X_predictions = []
        # for X_instance, y_instance in zip(X_test, y_test):
        #     predictions = []
        #     for estimator_ in self.fit_model.estimators_:
        #         predictions.append(estimator_.predict(X_instance.reshape(1, -1)))

        #     correct_votes = (np.array(predictions) == y_instance).sum()

        #     incorrect_votes = []
        #     for label in np.delete(np.unique(y_test), np.where(np.unique(y_test) == y_instance)):
        #         incorrect_votes.append((np.array(predictions) == label).sum())

        #     margin = (correct_votes - np.max(incorrect_votes)) / self.model.n_estimators

        #     margins.append(margin)
        #     X_predictions.append(y_instance if margin > 0 else 1 - y_instance)

        # return margins, X_predictions