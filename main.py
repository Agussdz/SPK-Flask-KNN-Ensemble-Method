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


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
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
        )
    return render_template("index.html", prediksi=None, probabilitas=None)


if __name__ == "__main__":
    app.run(debug=True)
