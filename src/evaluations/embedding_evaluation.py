import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_squared_error, accuracy_score
from npeet.entropy_estimators import entropy as calculate_entropy

from src.evaluations import BaseEvaluation


class EmbeddingsEvaluation(BaseEvaluation):
    def __init__(self, embedding_method_name, clustering_method_name, variance_threshold: float = 0.95, to_predict:str = 'Hour', max_iter:int = 50000, C: float = 0.1,
                 gamma:float = 0.01, kernel:str = 'linear' ):
        """
                Initialize the EmbeddingsEvaluation with parameters for PCA and SVC.
                Parameters:
                - variance_threshold: float, the cumulative variance threshold for PCA.
                - max_iter: int, maximum iterations for the SVC.
                - C: float, regularization parameter for the SVC.
                - gamma: float, kernel coefficient for the SVC.
                - kernel: str, type of kernel to be used in SVC.
                """
        super().__init__(embedding_method_name, clustering_method_name)
        self.variance_threshold = variance_threshold # For explained variance
        self.to_predict = to_predict
        self.model = SVC(random_state=42, max_iter=max_iter, C=C, gamma=gamma, kernel=kernel) # For classifier

    def evaluate_classifier(self, embeddings):
        embeddings['Time'] = embeddings.index.map(self.extract_hour)
        embeddings['Location'] = embeddings.index.map(self.extract_location)
        X = embeddings.drop(['Time', 'Location'], axis=1)
        y_time = embeddings['Time']
        y_location = embeddings['Location']

        X_train, X_test, y_time_train, y_time_test = train_test_split(X, y_time, test_size=0.2, random_state=42)
        _, _, y_location_train, y_location_test = train_test_split(X, y_location, test_size=0.2, random_state=42)

        svc_time = self.model.fit(X_train, y_time_train)
        svc_location = self.model.fit(X_train, y_location_train)

        y_time_pred = svc_time.predict(X_test)
        y_location_pred = svc_location.predict(X_test)

        results = {
            'f1_score_time': f1_score(y_time_test, y_time_pred, average='weighted'),
            'f1_score_location': f1_score(y_location_test, y_location_pred, average='weighted'),
            'accuracy_time': accuracy_score(y_time_test, y_time_pred),
            'accuracy_location': accuracy_score(y_location_test, y_location_pred)
        }
        return results

    def calculate_entropy(self, data):
        return calculate_entropy(data)

    def calculate_explained_variance(self, data):
        pca = PCA(n_components=None)  # Use all components
        pca.fit(data)
        pca_data = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(pca_data)
        num_components = np.argmax(cumulative_variance >= self.variance_threshold) + 1
        percentage_of_components = (num_components / len(pca_data)) * 100
        return percentage_of_components


    def evaluate(self):

        embeddings, _ = self.load_data()
        if not embeddings.empty:
            self.scaled_data = self.scale_data(embeddings)
            entropy_result = self.calculate_entropy(self.scaled_data)
            explained_variance_result = self.calculate_explained_variance(self.scaled_data)
            classifier_results = self.evaluate_classifier(embeddings)
            evaluation_results = {
                "Entropy": entropy_result,
                "Explained Variance": explained_variance_result,
                **classifier_results,
            }
        else:
            evaluation_results = {
                "Entropy": 0.0,
                "Explained Variance": 0.0,
                # **classifier_results
            }
        self.save_results('embeddings', evaluation_results)
        return
