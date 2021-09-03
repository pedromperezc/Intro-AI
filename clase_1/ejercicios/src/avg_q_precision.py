import numpy as np
from clase_1.ejercicios.src.basic_metrics_v2 import BaseMetric


class QueryMeanPrecision(BaseMetric):

    def __init__(self, **kwargs):
        BaseMetric.__init__(self, **kwargs)

    def __call__(self):

        """
        Definición:
        Mean Precision obtiene la media de la precisión de cada query. La precisión de una query es la cantidad de
        documentos 'true positive' (realmente relevantes), dividido por la cantidad de documentos obtenidos. Una
        precisión de 1 significa una precisión perfecta para la query.
        """

        truth_relevance = self.parameters["truth_relevance"]
        query_ids = self.parameters["q_id"]
        count_queries = np.bincount(query_ids)[1:]
        true_relevance_mask = (truth_relevance == 1)
        count_truth_relevance = np.bincount(query_ids[true_relevance_mask])[1:]

        return sum(count_truth_relevance / count_queries)/ len(count_queries)
