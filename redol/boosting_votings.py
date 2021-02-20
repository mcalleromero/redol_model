import numpy as np

from sklearn.ensemble import GradientBoostingClassifier


class GradientBoostingVotings:
    def __init__(self, n_estimators=100):
        self.model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=10)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def get_margins(self, X_test, y_test):
        """This function gets the margins for each instance of the Gradient Boosting Classifier

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
