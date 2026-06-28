import pickle

from sklearn.ensemble import RandomForestClassifier


class Trainer:

    def train(self, X_train, y_train):

        model = RandomForestClassifier(
            random_state=42
        )

        model.fit(X_train, y_train)

        pickle.dump(
            model,
            open("credit_model.pkl","wb")
        )

        return model