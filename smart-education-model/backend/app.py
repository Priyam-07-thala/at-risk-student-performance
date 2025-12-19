from flask import Flask, render_template, request, redirect, session
import sqlite3, os, pandas as pd, joblib

app = Flask(__name__)
app.secret_key = "secret123"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database.db")

# ---------- DATABASE ----------
def get_db():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        role TEXT,
        student_id TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS students(
        student_id TEXT PRIMARY KEY,
        name TEXT,
        attendance REAL,
        avg_marks REAL,
        assignment_completion REAL,
        behavior_score REAL,
        risk TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()

# ---------- ML ----------
bundle = joblib.load(os.path.join(BASE_DIR, "ml", "model.pkl"))
model = bundle["model"]
le = bundle["label_encoder"]

# ---------- ROUTES ----------
@app.route("/")
def role_select():
    return render_template("role_select.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO users(username,password,role,student_id)
        VALUES (?,?,?,?)
        """, (
            request.form["username"],
            request.form["password"],
            request.form["role"],
            request.form.get("student_id")
        ))
        conn.commit()
        conn.close()
        return redirect("/login")
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
        SELECT * FROM users WHERE username=? AND password=?
        """, (request.form["username"], request.form["password"]))
        user = cur.fetchone()
        conn.close()

        if user:
            session["role"] = user[3]
            session["student_id"] = user[4]
            return redirect("/teacher" if user[3]=="teacher" else "/student")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

@app.route("/teacher")
def teacher_dashboard():
    if session.get("role") != "teacher":
        return redirect("/login")

    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
    SELECT 
    AVG(attendance),
    AVG(avg_marks),
    AVG(assignment_completion),
    AVG(behavior_score)
    FROM students
    """)
    avg = cur.fetchone()

    chart_data = [avg[0] or 0, avg[1] or 0, avg[2] or 0, avg[3] or 0]

    return render_template(
        "teacher_dashboard.html",
        students='students',
        chart_data=chart_data
    )


@app.route("/student")
def student_dashboard():
    if session.get("role") != "student":
        return redirect("/login")

    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT attendance, avg_marks, assignment_completion, behavior_score, risk FROM students WHERE student_id=?",
        (session["student_id"],)
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        return render_template("student_dashboard.html", student=None)

    student_data = {
        "attendance": row[0],
        "marks": row[1],
        "assignments": row[2],
        "behavior": row[3],
        "risk": row[4]
    }

    return render_template(
        "student_dashboard.html",
        student_data=student_data
    )

@app.route("/upload", methods=["POST"])
def upload_csv():
    if session.get("role") != "teacher":
        return redirect("/login")

    df = pd.read_csv(request.files["csv_file"])

    conn = get_db()
    cur = conn.cursor()

    for _, row in df.iterrows():
        X = pd.DataFrame([{
            "attendance": row["attendance"],
            "avg_marks": row["avg_marks"],
            "assignment_completion": row["assignment_completion"],
            "behavior_score": row["behavior_score"]
        }])

        pred = model.predict(X)
        risk = le.inverse_transform(pred)[0]

        cur.execute("""
        INSERT OR REPLACE INTO students
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            row["student_id"], row["name"],
            row["attendance"], row["avg_marks"],
            row["assignment_completion"],
            row["behavior_score"], risk
        ))

    conn.commit()
    conn.close()
    return redirect("/teacher")

if __name__ == "__main__":
    app.run(debug=True)
