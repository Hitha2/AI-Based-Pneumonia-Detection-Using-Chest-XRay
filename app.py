import os
import bcrypt
import sqlite3
import numpy as np
from flask import Flask, render_template, request, redirect, session, flash, send_from_directory, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image
import base64
from flask import make_response

# ---------------- App Setup ----------------
app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DB_FILE = "patients.db"

# ---------------- Load ML Model ----------------
model = load_model("models/pneumonia_model.h5")

# ---------------- Dummy Doctor Login ----------------
#stored_password = bcrypt.hashpw("doctor123".encode(), bcrypt.gensalt())

# ---------------- Database Init ----------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            gender TEXT,
            image TEXT,
            result TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Prescriptions table
    c.execute("""
        CREATE TABLE IF NOT EXISTS prescriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            medicines TEXT,
            advice TEXT,
            followup TEXT,
            prescribed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(id)
        )
    """)
    
    # USERS TABLE ✅
    #c.execute("""
        #CREATE TABLE IF NOT EXISTS users (
           # id INTEGER PRIMARY KEY AUTOINCREMENT,
         #   username TEXT UNIQUE NOT NULL,
           # email TEXT UNIQUE NOT NULL,
           # password TEXT NOT NULL
       # )
    ""#")

    conn.commit()
    conn.close()

init_db()

# ---------------- DB Helpers ----------------
def save_patient(data):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        INSERT INTO patients (name, age, gender, image, result)
        VALUES (?, ?, ?, ?, ?)
    """, (data["name"], data["age"], data["gender"], data["image"], data["result"]))
    conn.commit()
    conn.close()

def get_all_patients():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM patients ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    return rows

# ---------------- Prediction Helper ----------------
def predict_image(image_path):
    img = Image.open(image_path).resize((224, 224)).convert("RGB")
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    prob = model.predict(img_array)[0][0]

    if prob > 0.55:
        return "PNEUMONIA", round(prob * 100, 2)
    else:
        return "NORMAL", round((1 - prob) * 100, 2)

def image_to_base64(image_path):
    with open(image_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

# Dummy user storage (in memory)
users = {}  # format: {username: password}

# ---------------- Routes ----------------

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()

        c.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username, password)
        )
        user = c.fetchone()
        conn.close()

        if user:
            session["username"] = username
            session["logged"] = True
            flash("Login successful!")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials!")
            return redirect(url_for("login"))

    return render_template("login.html")


# Dashboard route
@app.route("/dashboard")
def dashboard():
    if not session.get("logged"):
        return redirect(url_for("login"))
    patients = get_all_patients()
    return render_template("dashboard.html", patients=patients)




@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()

        try:
            c.execute(
                "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                (username, email, password)
            )
            conn.commit()
            flash("Registration successful! You can now login.")
            return redirect(url_for("login"))

        except sqlite3.IntegrityError:
            flash("Username or Email already exists!")
            return redirect(url_for("register"))

        finally:
            conn.close()

    return render_template("register.html")




#@app.route("/dashboard")
#def dashboard():
    #if not session.get("logged"):
     #   return redirect(url_for("login"))
    #patients = get_all_patients()
    #return render_template("dashboard.html", patients=patients)

@app.route("/add-patient", methods=["GET", "POST"])
def add_patient():
    if not session.get("logged"):
        return redirect(url_for("login"))

    if request.method == "POST":
        name = request.form["name"]
        age = request.form["age"]
        gender = request.form["gender"]
        file = request.files["xray"]

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        label, confidence = predict_image(filepath)

        save_patient({
            "name": name,
            "age": age,
            "gender": gender,
            "image": filename,
            "result": f"{label} ({confidence}%)"
        })

        return render_template(
            "result.html",
            name=name,
            result=label,
            confidence=confidence,
            image=f"{UPLOAD_FOLDER}/{filename}"
        )

    return render_template("patient_form.html")

@app.route("/history")
def patient_history():
    if not session.get("logged"):
        return redirect(url_for("login"))
    patients = get_all_patients()
    return render_template("history.html", patients=patients)

# ---------------- EDIT PATIENT ----------------
@app.route("/edit/<int:id>", methods=["GET", "POST"])
def edit_patient(id):
    if not session.get("logged"):
        return redirect(url_for("login"))

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    if request.method == "POST":
        name = request.form["name"]
        age = request.form["age"]
        gender = request.form["gender"]

        c.execute("""
            UPDATE patients
            SET name=?, age=?, gender=?
            WHERE id=?
        """, (name, age, gender, id))
        conn.commit()
        conn.close()

        flash("Patient updated successfully")
        return redirect(url_for("patient_history"))

    c.execute("SELECT id, name, age, gender FROM patients WHERE id=?", (id,))
    patient = c.fetchone()
    conn.close()

    return render_template("edit_patient.html", patient=patient)

# ---------------- DELETE PATIENT ----------------
@app.route("/delete/<int:id>")
def delete_patient(id):
    if not session.get("logged"):
        return redirect(url_for("login"))

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("SELECT image FROM patients WHERE id=?", (id,))
    record = c.fetchone()

    if record:
        image_path = os.path.join(UPLOAD_FOLDER, record[0])
        if os.path.exists(image_path):
            os.remove(image_path)

    c.execute("DELETE FROM patients WHERE id=?", (id,))
    conn.commit()
    conn.close()

    flash("Patient deleted successfully")
    return redirect(url_for("patient_history"))

@app.route("/download/<filename>")
def download_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/report/<int:id>")
def view_report(id):
    if not session.get("logged"):
        return redirect(url_for("login"))

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM patients WHERE id=?", (id,))
    patient = c.fetchone()
    conn.close()

    image_path = os.path.join(UPLOAD_FOLDER, patient[4])
    image_base64 = image_to_base64(image_path)

    return render_template(
        "report.html",
        patient=patient,
        image_base64=image_base64
    )


@app.route("/download-report/<int:id>")
def download_report(id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM patients WHERE id=?", (id,))
    patient = c.fetchone()
    conn.close()

    image_path = os.path.join(UPLOAD_FOLDER, patient[4])
    image_base64 = image_to_base64(image_path)

    rendered = render_template(
        "report.html",
        patient=patient,
        image_base64=image_base64,
        download=True
    )

    response = make_response(rendered)
    response.headers["Content-Disposition"] = "attachment; filename=patient_report.html"
    return response

@app.route("/view-prescription/<int:id>")
def view_prescription(id):
    if not session.get("logged"):
        return redirect(url_for("login"))

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
        SELECT p.name, p.age, p.gender,
               pr.medicines, pr.advice, pr.followup, pr.prescribed_at
        FROM patients p
        JOIN prescriptions pr ON p.id = pr.patient_id
        WHERE p.id=?
        ORDER BY pr.prescribed_at DESC
        LIMIT 1
    """, (id,))

    data = c.fetchone()
    conn.close()

    return render_template("view_prescription.html", data=data)

@app.route("/prescription/<int:id>", methods=["GET", "POST"])
def prescription(id):
    if not session.get("logged"):
        return redirect(url_for("login"))

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("SELECT * FROM patients WHERE id=?", (id,))
    patient = c.fetchone()

    if request.method == "POST":
        medicines = request.form["medicines"]
        advice = request.form["advice"]
        followup = request.form["followup"]

        c.execute("""
            INSERT INTO prescriptions (patient_id, medicines, advice, followup)
            VALUES (?, ?, ?, ?)
        """, (id, medicines, advice, followup))

        conn.commit()
        conn.close()

        flash("Prescription saved successfully")
        return redirect(url_for("view_prescription", id=id))

    conn.close()
    return render_template("prescription.html", patient=patient)


@app.route("/download-prescription/<int:patient_id>")
def download_prescription(patient_id):
    if not session.get("logged"):
        return redirect(url_for("login"))

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("SELECT * FROM patients WHERE id=?", (patient_id,))
    patient = c.fetchone()

    c.execute("SELECT * FROM prescriptions WHERE patient_id=?", (patient_id,))
    prescription = c.fetchone()

    conn.close()

    rendered = render_template(
        "view_prescription.html",
        patient=patient,
        prescription=prescription
    )

    response = make_response(rendered)
    response.headers["Content-Disposition"] = (
        f"attachment; filename=prescription_{patient[1]}.html"
    )

    return response




# ---------------- Run App ----------------
if __name__ == "__main__":
    app.run(debug=True)
