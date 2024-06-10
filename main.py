<<<<<<< HEAD
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
=======
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

#tess
app = Flask(__name__)

# Load data
df = pd.read_excel("Data_train.xlsx")

# Preprocess data
x = df[["K1", "K2", "K3", "K4", "K5"]]
y = df["Hasil"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=31
)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Define weighted euclidean distance
weights = np.array([0.2, 0.3, 0.5, 0.1, 0.2])


def weighted_euclidean_distance(x, y):
    return np.sqrt(np.sum(weights * (x - y) ** 2))


# Train the model
base_estimator = KNeighborsClassifier(n_neighbors=7, metric=weighted_euclidean_distance)
classifier = BaggingClassifier(base_estimator, n_estimators=10, random_state=42)
classifier.fit(x_train, y_train)


# Function to make prediction
def predict_credit(x_input):
    input_user = sc.transform([x_input])
    prediksi = classifier.predict(input_user)
    probabilitas = classifier.predict_proba(input_user)[0]
    return prediksi[0], probabilitas[1] * 100
>>>>>>> 9ef2c6b1057ffedbcf08aaefc261ded3854a148d


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
<<<<<<< HEAD
        input_user = []
        for i in range(5):
            input_user.append(int(request.form["K{}".format(i + 1)]))

        acceptance_prob = predict_acceptance_prob(input_user)
        prediksi = 1 if acceptance_prob > 50 else 0

        return render_template(
            "index.html", prediksi=prediksi, probabilitas=acceptance_prob
=======
        # Get user input
        kriteria = [
            int(request.form["K1"]),
            int(request.form["K2"]),
            int(request.form["K3"]),
            int(request.form["K4"]),
            int(request.form["K5"]),
        ]
        # Make prediction
        hasil_prediksi, probabilitas = predict_credit(kriteria)
        return render_template(
            "index.html", prediksi=hasil_prediksi, probabilitas=probabilitas
>>>>>>> 9ef2c6b1057ffedbcf08aaefc261ded3854a148d
        )
    return render_template("index.html", prediksi=None, probabilitas=None)


if __name__ == "__main__":
    app.run(debug=True)
