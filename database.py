import sqlite3

DB_FILE = "patients.db"

def save_patient_db(data):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        INSERT INTO patients (name, age, gender, symptoms, image, result)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (data['name'], data['age'], data['gender'], data.get('symptoms', ''), data['image'], data['result']))
    conn.commit()
    conn.close()

def get_all_patients():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, name, age, gender, symptoms, image, result, created_at FROM patients ORDER BY created_at DESC")
    patients = c.fetchall()
    conn.close()
    return patients
