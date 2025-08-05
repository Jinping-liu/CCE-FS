# -*- encoding: utf-8 -*-
import copy
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from pandas import Categorical
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor as LOF
from sklearn.preprocessing import LabelEncoder


class GeneratePopulationWithKDTree:
    def __init__(self, pred_model, dataset, population_size, params, pipeline_path):
        self.__labelencoders = None
        self.pred_model = pred_model
        self.population_size = population_size
        self.dataset = dataset
        self.pipeline_path = pipeline_path
        self.pipeline = None

        if isinstance(params, dict):
            for k, v in params.items():
                setattr(self, k, v)
        if not self.one_hot_train_sequence == self.one_hot_test_sequence:
            raise Exception("column sequence error !!!")
        if self.target_name in self.one_hot_train_sequence:
            self.one_hot_train_sequence.remove(self.target_name)

        # init label encoder
        if not self.__labelencoders:
            self.fit_label_encoders()
            self.get_valid_feature_range()

        self.load_pipline()

    def load_pipline(self):
        if not self.pipeline:
            self.pipeline = joblib.load(self.pipeline_path)

    def set_feature_dtypes_and_dummies(self, data):
        _data = self._set_feature_dtypes(data)
        return pd.get_dummies(_data)

    def set_category_with_query_instance(self, query_instance):
        for i in self.category_feature_names:
            cat = Categorical(self.feature_range[i])
            query_instance[i] = query_instance[i].astype(cat)

    def get_dataset_dummies(self):
        y = self.dataset[self.target_name]
        nor_arr = self.pipeline.transform(self.dataset.drop(self.target_name, axis=1))
        X = pd.DataFrame(nor_arr, columns=self.one_hot_train_sequence, index=self.dataset.index)
        X[self.target_name] = y

        self.dataset_copy_with_dummies = X

        # 计算经过标准化处理后的精度，用于后置的post fix
        self.get_decimal_precisions()

    def get_query_instance_dummies(self, query_instance):
        nor_arr = self.pipeline.transform(query_instance)
        query_instance_dummies = pd.DataFrame(nor_arr, columns=self.one_hot_train_sequence)

        return query_instance_dummies

    def build_tree(self, dummies):
        KD_tree = KDTree(dummies)
        return KD_tree

    def build_KD_tree(self, query_instance, query_instance_target, normalize: bool = True):
        if not hasattr(self, "dataset_copy_with_dummies"):
            self.get_dataset_dummies()

        predictions = self.pred_model.predict(
            self.dataset_copy_with_dummies.drop(columns=self.target_name).values).astype(
            self.dataset[self.target_name].dtype)
        self.dataset["pred_" + self.target_name] = predictions

        query_instance_dummies = self.get_query_instance_dummies(query_instance)
        print(f"query instance predict target:{self.pred_model.predict(query_instance_dummies.values)[0]}")

        self.query_instance_output_nodes = 2

        # segmenting the dataset according to outcome
        self.desired_class = query_instance_target
        dataset_with_predictions = self.dataset.loc[predictions == query_instance_target].copy()

        dummies = self.get_query_instance_dummies(dataset_with_predictions[self.feature_names])
        KD_tree = self.build_tree(dummies)

        kmeans = None
        if hasattr(self, "build_kmeans"):
            kmeans = self.build_kmeans(dummies.values)

        self.lof = LOF(n_neighbors=20, novelty=True, n_jobs=-1)
        self.lof.fit(dummies.values)

        return dataset_with_predictions, KD_tree, kmeans, predictions, query_instance_dummies

    @lru_cache()
    def get_feature_index(self):
        cont = []
        cat = []
        for name in self.one_hot_train_sequence:
            if name in self.continuous_feature_names:
                cont.append(self.one_hot_index.get(name))
            else:
                cat.append(self.one_hot_index.get(name))
        return cont, cat

    def reverse_transform(self, df_transformed):
        cont_index, cat_index = self.get_feature_index()
        new_array = np.empty((df_transformed.shape[0], len(self.feature_names)), dtype=object)

        for name, transformer in self.pipeline.named_transformers_.items():
            if name == "num":
                selected_arr = df_transformed[:, cont_index]
                inversed_cont_features = transformer.inverse_transform(selected_arr)
                for feature_name, idx in zip(self.continuous_feature_names, cont_index):
                    _type = self.feature_dtypes.get(feature_name)
                    new_array[:, idx] = inversed_cont_features[:, idx].astype(_type)
            else:
                selected_arr = df_transformed[:, cat_index]
                new_array[:, cat_index[:len(self.feature_names) - len(
                    self.continuous_feature_names)]] = transformer.inverse_transform(selected_arr)

        return pd.DataFrame(new_array, columns=self.continuous_feature_names + self.category_feature_names)[
            self.feature_names]

    def single_transform(self, df_transformed, operate: str, is_return: bool = False):
        cont_index, cat_index = self.get_feature_index()
        transformer = self.pipeline.named_transformers_.get("num")

        if is_return:
            _df_transformed = copy.deepcopy(df_transformed)
            inversed_cont_features = transformer.transform(_df_transformed[:, cont_index])
            _df_transformed[:, cont_index] = inversed_cont_features
            return _df_transformed
        else:
            inversed_cont_features = transformer.transform(df_transformed[:, cont_index])
            df_transformed[:, cont_index] = inversed_cont_features


    def do_KD_tree_init(self, cfs, query_instance, cluster_centers=None, is_random=False):
        cfs = self.label_encode(cfs)
        cfs = cfs.reset_index(drop=True)
        query_instance_labeled = self.label_encode(query_instance.copy())

        if not is_random:
            count = int(self.population_size)
            self.cfs = np.zeros((count, len(self.feature_names)))
            for kx in range(count):
                if kx >= len(cfs):
                    break

                one_init = np.zeros(len(self.feature_names))
                for jx, feature in enumerate(self.feature_names):
                    if feature in self.features_not_to_vary:
                        one_init[jx] = (query_instance_labeled[feature])
                    else:
                        if feature in self.continuous_feature_names:
                            if self.feature_range[feature][0] <= cfs.iat[kx, jx] <= self.feature_range[feature][1]:
                                one_init[jx] = cfs.iat[kx, jx]
                            else:
                                if "float" in self.feature_dtypes[feature]:
                                    one_init[jx] = np.random.uniform(self.feature_range[feature][0],
                                                                     self.feature_range[feature][1])
                                else:
                                    one_init[jx] = np.random.randint(self.feature_range[feature][0],
                                                                     self.feature_range[feature][1] + 1)
                        else:
                            if cfs.iat[kx, jx] in self.feature_range[feature]:
                                one_init[jx] = cfs.iat[kx, jx]
                            else:
                                one_init[jx] = np.random.choice(self.feature_range[feature])

                self.cfs[kx] = one_init
                kx += 1

            uniques = np.unique(self.cfs, axis=0)

            if isinstance(cluster_centers, np.ndarray):
                num_individuals_per_cluster = int((self.population_size - len(uniques)) * 0.1) // self.n_clusters
                initial_population = [
                    center + np.random.normal(0, 0.01, size=center.shape)
                    for center in cluster_centers
                    for _ in range(num_individuals_per_cluster)
                ]

                if len(initial_population):
                    initial_population = self.reverse_transform(np.array(initial_population))
                    for feature in self.features_not_to_vary:
                        initial_population[feature] = query_instance[feature][0]
                    initial_population = self.label_encode(initial_population).astype('float64')
                    uniques = np.concatenate([uniques, initial_population], axis=0)

            if len(uniques) < self.population_size:
                remaining_cfs_count = len(uniques)
                remaining_cfs = self.do_random_init(self.population_size - remaining_cfs_count, query_instance,
                                                    query_instance_labeled)
                total_cfs = np.concatenate([uniques, remaining_cfs])
                uniques_random = np.unique(total_cfs, axis=0)
                self.cfs = uniques_random
            else:
                self.cfs = uniques
        else:
            remaining_cfs = self.do_random_init(self.population_size, query_instance, query_instance_labeled)
            uniques_random = np.unique(remaining_cfs, axis=0)
            self.cfs = uniques_random

    def get_all_dummy_colnames(self):
        return pd.get_dummies(self.dataset[self.feature_names]).columns

    def get_valid_feature_range(self):
        for name in self.category_feature_names:
            self.feature_range[name] = self.__labelencoders[name].transform(self.feature_range[name])

    def fix_columns(self, query_instance_df_dummies):
        """
        add missing items
        :param query_instance_df_dummies:
        :return:
        """
        df_copy = query_instance_df_dummies.copy()
        for col in self.one_hot_train_sequence:
            if col not in query_instance_df_dummies.columns:
                df_copy[col] = 0
        return df_copy

    def get_data_type(self, col):
        """Infers data type of a continuous feature from the training data."""
        if (self.dataset[col].dtype == np.int64) or (self.dataset[col].dtype == np.int32):
            return 'int'
        elif (self.dataset[col].dtype == np.float64) or (self.dataset[col].dtype == np.float32):
            return 'float'
        else:
            raise ValueError("Unknown data type of feature %s: must be int or float" % col)

    def _set_feature_dtypes(self, data_df):
        """Set the correct type of each feature column."""
        test = data_df.copy()
        test = test.reset_index(drop=True)

        if len(self.category_feature_names) > 0:
            for feature in self.category_feature_names:
                test[feature] = test[feature].apply(str)

            #  将标签数据转换为编码
            test[self.category_feature_names] = test[self.category_feature_names].astype('category')

        if len(self.continuous_feature_names) > 0:
            for feature in self.continuous_feature_names:
                if self.get_data_type(feature) == 'float':
                    test[feature] = test[feature].astype(np.float32)
                else:
                    test[feature] = test[feature].astype(np.int32)
        return test

    def get_decimal_precisions(self):
        """"Gets the precision of continuous features in the data."""
        # if the precision of a continuous feature is not given, we use the maximum precision of the modes to capture the precision of majority of values in the column.
        precisions = {}
        for ix, col in enumerate(self.continuous_feature_names):
            modes = self.dataset_copy_with_dummies[col].mode()
            precisions[col] = abs(modes[0] * 0.001)
        self.precisions = precisions

    def fit_label_encoders(self):
        labelencoders = {}
        labelencoder_mapping = {}
        for column in self.category_feature_names:
            labelencoders[column] = LabelEncoder()
            labelencoders[column] = labelencoders[column].fit(self.dataset[column])
            mapping = dict(
                zip(labelencoders[column].classes_, labelencoders[column].transform(labelencoders[column].classes_)))
            labelencoder_mapping[column] = mapping

        self.labelencoder_mapping = labelencoder_mapping
        self.__labelencoders = labelencoders

    def label_encode(self, input_instance):
        for c in self.category_feature_names:
            input_instance[c] = self.__labelencoders[c].transform(input_instance[c])

        return input_instance

    def label_decode(self, labelled_input):
        """Transforms label encoded data back to categorical values
        """
        num_to_decode = 1
        if len(labelled_input.shape) > 1:
            num_to_decode = len(labelled_input)
        else:
            labelled_input = [labelled_input]

        input_instance = []

        for j in range(num_to_decode):
            temp = {}
            for i in range(len(labelled_input[j])):
                if self.feature_names[i] in self.category_feature_names:
                    enc = self.__labelencoders[self.feature_names[i]]
                    val = enc.inverse_transform(np.array([labelled_input[j][i]], dtype=np.int32))
                    temp[self.feature_names[i]] = val[0]
                else:
                    temp[self.feature_names[i]] = labelled_input[j][i]
            input_instance.append(temp)
        input_instance_df = pd.DataFrame(input_instance, columns=self.feature_names)
        return input_instance_df

    @staticmethod
    def check_column_sequence(data_set1, data_set2):
        return data_set1 == data_set2

    def do_random_init(self, num_inits, query_instance, query_instance_labeled):
        remaining_cfs = np.zeros((num_inits, len(self.feature_names)))

        kx = 0
        while kx < num_inits:
            one_init = np.zeros(len(self.feature_names))
            for jx, feature in enumerate(self.feature_names):
                if feature in self.features_to_vary:
                    if feature in self.continuous_feature_names:
                        if self.feature_dtypes[feature] in ['int64', 'int32', 'int16', 'int8']:
                            one_init[jx] = np.random.randint(self.feature_range[feature][0],
                                                             self.feature_range[feature][1] + 1)
                        else:
                            one_init[jx] = np.random.uniform(self.feature_range[feature][0],
                                                             self.feature_range[feature][1])
                    else:
                        one_init[jx] = np.random.choice(self.feature_range[feature]).astype(
                            self.feature_dtypes[feature])
                else:
                    one_init[jx] = query_instance_labeled[feature]
            remaining_cfs[kx] = one_init
            kx += 1
        return remaining_cfs

    def build_population(self, query_instance, query_instance_target, is_one_hot: bool = False):
        dataset_with_predictions, KD_tree, _, _, query_instance_dummies = self.build_KD_tree(query_instance,
                                                                                             query_instance_target)
        num_queries = min(len(dataset_with_predictions), self.population_size)
        indices = KD_tree.query(query_instance_dummies, num_queries)[1][0]
        kD_tree_output = dataset_with_predictions.iloc[indices].copy()
        self.do_KD_tree_init(kD_tree_output, query_instance)

        if not is_one_hot:
            return self.cfs
        else:
            # 将数据转为one-hot编码
            cfs = self.label_decode(self.cfs)
            cfs_dummies = self.get_query_instance_dummies(cfs)
            return cfs_dummies


class GeneratePopulationWithKmeans(GeneratePopulationWithKDTree):
    def __init__(self, pred_model, dataset, population_size, params, n_clusters, random_state, pipeline_path):
        super().__init__(pred_model, dataset, population_size, params, pipeline_path)

        self.n_clusters = n_clusters
        self.random_state = random_state

    def build_kmeans(self, dummies):
        self.infer_kmeans_elbow(X=dummies, k=self.n_clusters, random_state=self.random_state)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        _ = kmeans.fit_predict(dummies)
        return kmeans.cluster_centers_

    def infer_kmeans_elbow(self, X=None, random_state=0, k=20, is_show=False):
        inertia = []
        for i in range(2, k + 1):
            kmeans = KMeans(n_clusters=i, random_state=random_state)
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)

        kneedle = KneeLocator(list(range(2, k + 1)), inertia, curve='convex', direction='decreasing')
        best_k = kneedle.elbow

        if best_k and is_show:
            plt.figure(figsize=(8, 6))
            plt.plot(range(2, k + 1), inertia, marker='o')
            plt.xlabel('Number of clusters (K)')
            plt.ylabel('Inertia (Within-cluster sum of squares)')
            plt.title('Elbow Method for Optimal K')
            plt.grid(True)

            plt.annotate(f'Best K = {best_k}', xy=(best_k, kneedle.elbow_y), xytext=(best_k, kneedle.elbow_y + 500),
                         arrowprops=dict(arrowstyle='->'))

            plt.show()
        else:
            if best_k is None:
                best_k = 6
        self.n_clusters = best_k

    def build_population(self, query_instance, query_instance_target, is_one_hot: bool = False,
                         is_random: bool = False):
        dataset_with_predictions, KD_tree, cluster_centers, _, query_instance_dummies = self.build_KD_tree(
            query_instance,
            query_instance_target)
        num_queries = min(len(dataset_with_predictions), self.population_size)
        indices = KD_tree.query(query_instance_dummies, num_queries)[1][0]
        kD_tree_output = dataset_with_predictions.iloc[indices].copy()
        self.do_KD_tree_init(kD_tree_output, query_instance, cluster_centers, is_random)

        if not is_one_hot:
            return self.cfs, query_instance_dummies
        else:
            # 将数据转为one-hot编码
            cfs = self.label_decode(self.cfs)
            cfs_dummies = self.fix_columns(pd.get_dummies(cfs))[self.one_hot_train_sequence]
            query_instance_dummies = self.fix_columns(pd.get_dummies(query_instance))[self.one_hot_train_sequence]
            return cfs_dummies, query_instance_dummies


if __name__ == '__main__':
    pass
