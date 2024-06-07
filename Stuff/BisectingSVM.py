from sklearn.svm import SVR
from itertools import product
from Bisectingkmedoids import BisectingKMedoidsGenerator, BisectingKMedoids

class BisectingSVM:
    def run_bisecting_svm(self, klaster_sets, c_values, gamma_values):
        bisecting = BisectingKMedoids()
        errors = []

        for c_value, gamma_value in product(c_values, gamma_values):
            for klaster_set in klaster_sets:
                raw_test_data = klaster_set['clusters'][0][0]  # Mengambil data uji dari klaster pertama
                test_data = raw_test_data['actual']
                actual_y = [tupel['actual'] for cluster in klaster_set['clusters'] for tupel in cluster]
                data_nn = [self.converting_ecf_to_nn(tuples, 1) for cluster in klaster_set['clusters'] for tuples in cluster]

                svr = SVR(kernel='rbf', C=c_value, gamma=gamma_value)
                svr.fit(data_nn, actual_y)
                estimated_effort = svr.predict([test_data])

                ae = abs(estimated_effort - raw_test_data['actual'])
                errors.append(ae)

        mae = sum(errors) / (len(c_values) * len(gamma_values) * len(klaster_sets))
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
