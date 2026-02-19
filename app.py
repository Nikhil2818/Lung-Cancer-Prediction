


from flask import Flask, render_template, request
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# ===============================
# Load Model and Dummy Columns
# ===============================
model = joblib.load("rsf_model (6).pkl")
dummy_columns = joblib.load("dummy_columns.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # -------- Collect Input --------
        input_data = {
            "Age_group": int(request.form["Age_group"]),
            "Sex_binary": int(request.form["Sex_binary"]),
            "Laterality_group": int(request.form["Laterality_group"]),
            "Stage_group": int(request.form["Stage_group"]),
            "T_group": int(request.form["T_group"]),
            "N_group": int(request.form["N_group"]),
            "TumorSize_group": int(request.form["TumorSize_group"]),
            "Nodes_examined_group": int(request.form["Nodes_examined_group"]),
            "Nodes_positive_group": int(request.form["Nodes_positive_group"]),
            "Surgery_group": int(request.form["Surgery_group"]),
            "Chemo_binary": int(request.form["Chemo_binary"]),
            "PrimarySite_group": int(request.form["PrimarySite_group"]),
            "TumorType_group": int(request.form["TumorType_group"]),
            "grade_num": int(request.form["grade_num"]),
        }

        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df, drop_first=True)
        input_df = input_df.reindex(columns=dummy_columns, fill_value=0)

        # -------- Predict Survival Function --------
        surv_fn = model.predict_survival_function(input_df)
        surv = surv_fn[0]

        # -------- Probability Function (MONTHS) --------
        def get_prob(year):
            months = year * 12
            max_time = surv.domain[1]
            if months > max_time:
                months = max_time
            return float(surv(months))

        p1 = round(get_prob(1), 3)
        p3 = round(get_prob(3), 3)
        p5 = round(get_prob(5), 3)

        # -------- Plot Survival Curve --------
        times = surv.x
        probs = surv.y

        plt.figure()
        plt.plot(times, probs)
        plt.xlabel("Time (Months)")
        plt.ylabel("Survival Probability")
        plt.title("Predicted Survival Curve")
        plt.ylim(0, 1)

        img = io.BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return render_template(
            "index.html",
            prediction_text={
                "1_year": p1,
                "3_year": p3,
                "5_year": p5
            },
            plot_url=plot_url
        )

    except Exception as e:
        return render_template(
            "index.html", prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)
