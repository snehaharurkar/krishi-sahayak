from flask import Flask, render_template, request
import os
import numpy as np
import tensorflow as tf
import base64

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- Load your model ----------------
disease_model = tf.keras.models.load_model("rice_disease_model.h5")  # Replace with your model file path

# ---------------- Prediction function ----------------
def predict_disease(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = disease_model.predict(img_array)[0]
    class_index = np.argmax(predictions)
    confidence = float(np.max(predictions)) * 100

    class_labels = {
        0: "Bacterial Leaf Blight",
        1: "Brown Spot",
        2: "Healthy",
        3: "Leaf Blast"
    }

    return class_labels[class_index], round(confidence, 2)

# ---------------- Routes ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":
        # ---------- IMAGE UPLOAD ----------
        if "file" in request.files:
            file = request.files["file"]
            if file.filename != "":
                image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                file.save(image_path)
                prediction, confidence = predict_disease(image_path)

        # ---------- CAMERA ----------
        elif "camera_image" in request.form:
            data_url = request.form["camera_image"]
            if data_url:
                header, encoded = data_url.split(",", 1)
                data = base64.b64decode(encoded)
                image_path = os.path.join(app.config["UPLOAD_FOLDER"], "camera.jpg")
                with open(image_path, "wb") as f:
                    f.write(data)
                prediction, confidence = predict_disease(image_path)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path
    )

if __name__ == "__main__":
    app.run(debug=True)
