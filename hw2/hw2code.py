import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    if len(feature_vector) < 2:
        return [], [], None, None

    sorted_idx = np.argsort(feature_vector)
    sorted_f = feature_vector[sorted_idx]
    sorted_t = target_vector[sorted_idx]
    n = len(sorted_f)

    diff = sorted_f[1:] > sorted_f[:-1]
    split_indices = np.where(diff)[0] + 1

    if len(split_indices) == 0:
        return [], [], None, None

    thresholds = (sorted_f[split_indices - 1] + sorted_f[split_indices]) / 2.0

    cum_ones = np.cumsum(sorted_t)
    ones_left = cum_ones[split_indices - 1]
    left_sizes = split_indices.astype(float)
    p1_l = ones_left / left_sizes
    p0_l = 1 - p1_l
    H_l = 1 - p1_l**2 - p0_l**2

    total_ones = cum_ones[-1]
    ones_right = total_ones - ones_left
    right_sizes = n - left_sizes
    p1_r = ones_right / right_sizes
    p0_r = 1 - p1_r
    H_r = 1 - p1_r**2 - p0_r**2

    ginis = - (left_sizes / n) * H_l - (right_sizes / n) * H_r

    best_idx = int(np.argmax(ginis))
    threshold_best = float(thresholds[best_idx])
    gini_best = float(ginis[best_idx])

    return thresholds, ginis, threshold_best, gini_best

class DecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split if min_samples_split is not None else 2
        self._min_samples_leaf = min_samples_leaf if min_samples_leaf is not None else 1

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self._max_depth is not None and depth >= self._max_depth: # Дополнительные остановки
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if np.all(sub_y == sub_y[0]): # Исправлено: == вместо !=
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        best_categories_map = None  # Добавлено для сохранения map
        for feature in range(sub_X.shape[1]): # Исправлено: с 0 вместо 1
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    current_click = clicks.get(key, 0)
                    ratio[key] = current_click / current_count if current_count > 0 else 0 # Исправлено: click / count
                sorted_categories = [key for key, _ in sorted(ratio.items(), key=lambda x: x[1])] # Исправлено: x[0] вместо x[1]
                categories_map = {cat: i for i, cat in enumerate(sorted_categories)}
                feature_vector = np.array([categories_map.get(x, 0) for x in sub_X[:, feature]])
            else:
                raise ValueError
            if len(np.unique(feature_vector)) < 2:
                continue # Пропуск константных

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if threshold is None:
                continue

            split_temp = feature_vector < threshold
            left_size = np.sum(split_temp)
            if left_size < self._min_samples_leaf or (len(sub_y) - left_size) < self._min_samples_leaf: # Добавлена проверка
                continue

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = split_temp

                if feature_type == "real":
                    threshold_best = threshold
                    best_categories_map = None
                elif feature_type == "categorical": # Было "Categorical"
                    threshold_best =  threshold
                    best_categories_map = categories_map.copy()  # Добавлено: копия map
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0] # Исправлено: [0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            if best_categories_map is not None:
                node["categories_split"] = [k for k, v in best_categories_map.items() if v < threshold_best]
            else:
                node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"], depth + 1) # Исправлено: ~split для right

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        f = node["feature_split"]
        if self._feature_types[f] == "real":
            return self._predict_node(
                x,
                node["left_child"] if x[f] < node["threshold"] else node["right_child"],
            )
        else:
            cats_left = set(node["categories_split"])
            return self._predict_node(
                x, node["left_child"] if x[f] in cats_left else node["right_child"]
            )

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

    def get_params(self, deep=True):
        return {
            "feature_types": self._feature_types,
            "max_depth": self._max_depth,
            "min_samples_split": self._min_samples_split,
            "min_samples_leaf": self._min_samples_leaf,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)
        return self  # Добавлено для chainability