import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# Bobot untuk setiap opsi kriteria
bobot_k1 = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
bobot_k2 = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
bobot_k3 = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
bobot_k4 = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
bobot_k5 = np.array([0.10, 0.15, 0.20, 0.25, 0.30])


dataset_train = pd.read_excel("data_training.xlsx")
dataset_test = pd.read_excel("data_testing.xlsx")

# bagging
bagging_classifier = BaggingClassifier(
    estimator=KNeighborsClassifier(n_neighbors=9), n_estimators=10, random_state=42
)

# Latih model Bagging pada data training
X_train = dataset_train.iloc[:, :-1]  # Fitur
y_train = dataset_train.iloc[:, -1]  # Label
bagging_classifier.fit(X_train, y_train)


# Function untuk memprediksi probabilitas diterimanya (Hasil = 1)
def predict_acceptance_prob(input_user):
    jarak_list = []
    for index, row in dataset_train.iterrows():
        data_train = row[:-1].values
        jarak = np.sqrt(
            np.sum(
                [
                    bobot_k1[input_user[0] - 1]
                    * ((input_user[0] - data_train[0]) ** 2),
                    bobot_k2[input_user[1] - 1]
                    * ((input_user[1] - data_train[1]) ** 2),
                    bobot_k3[input_user[2] - 1]
                    * ((input_user[2] - data_train[2]) ** 2),
                    bobot_k4[input_user[3] - 1]
                    * ((input_user[3] - data_train[3]) ** 2),
                    bobot_k5[input_user[4] - 1]
                    * ((input_user[4] - data_train[4]) ** 2),
                ]
            )
        )
        jarak_list.append((index, jarak))

    # Mengurutkan berdasarkan jarak
    jarak_list_sorted = sorted(jarak_list, key=lambda x: x[1])

    # Mendapatkan 9 tetangga terdekat
    neighbors = jarak_list_sorted[:9]

    # Menghitung probabilitas diterimanya (Hasil = 1)
    acceptance_prob = (
        dataset_train.iloc[[idx[0] for idx in neighbors]]["Hasil"].mean() * 100
    )

    return acceptance_prob


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_user = []
        for i in range(5):
            input_user.append(int(request.form["K{}".format(i + 1)]))

        acceptance_prob = predict_acceptance_prob(input_user)
        prediksi = 1 if acceptance_prob > 50 else 0

        return render_template(
            "index.html", prediksi=prediksi, probabilitas=acceptance_prob
        )
    return render_template("index.html", prediksi=None, probabilitas=None)


if __name__ == "__main__":
    app.run(debug=True)
