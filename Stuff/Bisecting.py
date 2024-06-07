import numpy as np
import pandas as pd
import random
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error 
from itertools import product

class Dataset:
    def dataset(self):
        # Load dataset from CSV
        data = pd.read_csv("C:/Users/WIN10/OneDrive/Documents/Stuff/dataset.csv")
        # Convert dataset to list of dictionaries
        dataset = data.to_dict('records')
        return dataset
    
class MatrixLibrary:
    def inverse_matrix(self, matrix):
        try:
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            # Tangani jika matriks tidak dapat diubah
            return None

class BisectingKMedoids:
    def __init__(self):
        self.EXPONENT = 2

    def read_dataset(self, cacah):
        dataset = Dataset()
        temp = dataset.dataset()
        # Leave One Out - LOO
        ret = []
        for key, val in enumerate(temp):  
            if cacah != key:
                ret.append(val)
        return ret

    def get_test_data(self, cacah):
        dataset = Dataset()
        temp = dataset.dataset()
        # Leave One Out - LOO
        for key, val in enumerate(temp):
            if cacah == key:
                return val

    def random_medoid_tuple(self, arr_dataset):
        if not arr_dataset:
            return None
        random_index = random.randint(0, len(arr_dataset) - 1)
        return arr_dataset[random_index]

    def variance(self, arr_distance):
        if len(arr_distance) == 0:
            return 0
        else:
            return sum(arr_distance) / len(arr_distance)

    def distance(self, arr_dataset, arr_tuple_of_medoids):
        ret_distance = []
        for val in arr_dataset:
            f1 = (val['f1'] - arr_tuple_of_medoids['f1']) ** self.EXPONENT
            f2 = (val['f2'] - arr_tuple_of_medoids['f2']) ** self.EXPONENT
            f3 = (val['f3'] - arr_tuple_of_medoids['f3']) ** self.EXPONENT
            f4 = (val['f4'] - arr_tuple_of_medoids['f4']) ** self.EXPONENT
            f5 = (val['f5'] - arr_tuple_of_medoids['f5']) ** self.EXPONENT
            f6 = (val['f6'] - arr_tuple_of_medoids['f6']) ** self.EXPONENT
            f7 = (val['f7'] - arr_tuple_of_medoids['f7']) ** self.EXPONENT
            f8 = (val['f8'] - arr_tuple_of_medoids['f8']) ** self.EXPONENT
            ret_distance.append(f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8)
        return ret_distance

    def pairing_distance_and_project_id(self, arr_dataset, arr_distance):
        pairing = []
        for key, val in enumerate(arr_dataset):
            pairing.append({'projectID': val['project_id'], 'distance': arr_distance[key]})
        return pairing

    def child_variance(self, cluster):
        ret = {}
        for key, val in cluster.items():
            ret[key] = {}
            for subkey, subval in val.items():
                data_point = len(subval)
                cacah = sum(subsubval['distance'] for subsubval in subval)
                variance = cacah / data_point
                ret[key][subkey] = variance
        return ret

    def compare_variance(self, arr_parent_variance, arr_child_variance):
        ret = {}
        for key, val in arr_parent_variance.items():
            for subkey, subval in arr_child_variance.items():
                if key == subkey:
                    ret[key] = {}
                    for subsubkey, subsubval in subval.items():
                        if val < subsubval:
                            ret[key][subsubkey] = 'stop'
                        elif val > subsubval:
                            ret[key][subsubkey] = 'split'
        return ret

    def create_two_cluster(self, arr_distance_c1, arr_distance_c2):
        ret = {'C1': [], 'C2': []}
        for key, val in enumerate(arr_distance_c1):
            if val < arr_distance_c2[key]:
                ret['C1'].append(key)
            else:
                ret['C2'].append(key)
        return ret

    def converting_ecf(self, tuples):
        ret = []
        for key, val in tuples.items():
            if key != 'project_id' and key != 'actual' and key != 'size' and key != 'actualPF':
                ret.append(val)
        return ret

    def converting_ecf_to_nn(self, tuples, flag):
        ret = []
        for key, val in tuples.items():
            if (key == 'size' or key == 'actualPF') and flag == 0:
                ret.append(val)
            if key == 'actual' and flag == 1:
                ret.append(val)
        return ret

    def converting_raw(self, cluster, nn):
        converted_raws = []
        for key, tuples in cluster.items():
            if nn == 0:
                converted_raws.append(self.converting_ecf(tuples))
            if nn == 1:
                converted_raws.append(self.converting_ecf_to_nn(tuples, 0))
        return

    def bisecting_k_medoids_clustering(self, cacah):
        X = [self.read_dataset(cacah)]
        NextLevel = []
        V = X.copy()
        S = []
        arrMedoidForAllClusters = []

        while len(V) > 0:
            for val in V:
                arr_random_medoid_tuple_parent = self.random_medoid_tuple(val)
                parent_distance = self.distance(val, arr_random_medoid_tuple_parent)
                parent_variance = self.variance(parent_distance)
                arr_random_medoid_tuple_c1 = self.random_medoid_tuple(val)
                arr_random_medoid_tuple_c2 = self.random_medoid_tuple(val)
                arr_distance_c1 = self.distance(val, arr_random_medoid_tuple_c1)
                arr_distance_c2 = self.distance(val, arr_random_medoid_tuple_c2)
                variance_c1 = self.variance(arr_distance_c1)
                variance_c2 = self.variance(arr_distance_c2)
                two_cluster_c1_c2 = self.create_two_cluster(arr_distance_c1, arr_distance_c2)

                if max(variance_c1, variance_c2) < parent_variance:
                    for vals in two_cluster_c1_c2.values():
                        temp = [val[subval] for subval in vals]
                        NextLevel.append(temp)
                else:
                    S.append(val)
                    arrMedoidForAllClusters.append(arr_random_medoid_tuple_parent)

            V = NextLevel
            NextLevel = []

        ret = {'medoids': arrMedoidForAllClusters, 'clusters': S}
        return ret
    
class Utils:
    def transpose(self, data_nn):
        ret = []
        rett = []
        rettt = []
        for vals in data_nn:
            ret.append(vals[0])
            rett.append(vals[1])
            rettt.append(vals[2])
        rets = [ret, rett, rettt]
        return rets

    def sum_kuadrat(self, columns):
        kuadrat = [val ** 2 for val in columns]
        return sum(kuadrat)

    def multiplication(self, column1, column2):
        ret = [val1 * column2[key] for key, val1 in enumerate(column1)]
        return sum(ret)

    def matrix_multiplication(self, actual_y, transposed_data_nn):
        r0c0 = sum(transposed_data_nn[0])
        r0c1 = sum(transposed_data_nn[1])
        r0c2 = sum(transposed_data_nn[2])

        r1c0 = r0c1
        r1c1 = self.sum_kuadrat(transposed_data_nn[1])
        r1c2 = self.multiplication(transposed_data_nn[1], transposed_data_nn[2])

        r2c0 = r0c2
        r2c1 = r1c2
        r2c2 = self.sum_kuadrat(transposed_data_nn[2])

        x_transpose_x = [
            [r0c0, r0c1, r0c2],
            [r1c0, r1c1, r1c2],
            [r2c0, r2c1, r2c2]
        ]

        x_transpose_y0 = sum(actual_y)
        x_transpose_y1 = self.multiplication(actual_y, transposed_data_nn[1])
        x_transpose_y2 = self.multiplication(actual_y, transposed_data_nn[2])

        x_transpose_y = [x_transpose_y0, x_transpose_y1, x_transpose_y2]

        matrix_library = MatrixLibrary()
        x_transpose_x_inverse = matrix_library.inverse_matrix(x_transpose_x)

        w = [self.multiplication(x_transpose_y, vals) for vals in x_transpose_x_inverse]

        return w

class BisectingKMedoidsGenerator:
    def cluster_generator(self):
        cacah = 0
        bisecting = BisectingKMedoids()

        num_of_data = len(bisecting.read_dataset(0))

        klaster_sets = []
        for j in range(num_of_data):
            klasters = bisecting.bisecting_k_medoids_clustering(j)
            while cacah < 1:
                if len(klasters['clusters']) < 2:
                    klasters = bisecting.bisecting_k_medoids_clustering(j)
                    cacah = 0
                else:
                    break
            klaster_sets.append(klasters)

        return klaster_sets

class BisectingSVM:
    def optimize_svm(self, data, actual_y, c_values, gamma_values):
        best_mae = float('inf')
        best_model = None

        for c_value, gamma_value in product(c_values, gamma_values):
            svr = SVR(kernel='rbf', C=c_value, gamma=gamma_value)
            svr.fit(data, actual_y)
            predicted_y = svr.predict(data)
            mae = mean_absolute_error(actual_y, predicted_y)
            if mae < best_mae:
                best_mae = mae
                best_model = svr

        return best_model, best_mae 

    def run_bisecting_svm(self, klaster_sets, c_values, gamma_values):
        bisecting = BisectingKMedoids()
        errors = []

        for i, klaster_set in enumerate(klaster_sets):
            raw_test_data = bisecting.get_test_data(i)
            test_data = bisecting.converting_ecf(raw_test_data)
            actual_y = []
            raw_clusters = klaster_set['clusters']
            for cluster in raw_clusters:
                for tupel in cluster:
                    actual_y.append(tupel['actual'])

            data = []
            labels = []
            for key, klasters in enumerate(raw_clusters):
                for tupel in klasters:
                    labels.append(key)
                    data.append(bisecting.converting_ecf(tupel))

            # Optimize SVM for PF prediction
            svm_model, _ = self.optimize_svm(data, actual_y, c_values, gamma_values)

            # Predict PF using SVM
            predicted_pf = svm_model.predict([test_data])[0]

            # Calculate AE for PF prediction
            ae = abs(predicted_pf - raw_test_data['actualPF'])
            errors.append(ae)

        mae = sum(errors) / len(errors)
        return mae

if __name__ == "__main__":
    # Buat contoh objek dari kelas BisectingKMedoidsGenerator
    generator = BisectingKMedoidsGenerator()

    # Panggil metode cluster_generator untuk menghasilkan klaster sets
    klaster_sets = generator.cluster_generator()

    # Definisikan nilai-nilai C dan γ yang ingin dicoba
    c_values = [0.01, 100]
    gamma_values = [0.01, 50]

    # Buat contoh objek dari kelas BisectingSVM
    svm = BisectingSVM()

    # Jalankan metode run_bisecting_svm
    mae = svm.run_bisecting_svm(klaster_sets, c_values, gamma_values)

    # Tampilkan hasil Mean Absolute Error (MAE)
    print("Mean Absolute Error (MAE):", mae)
