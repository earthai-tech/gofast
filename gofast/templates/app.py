
"""
Install Flask: pip install flask
Place the HTML file in a folder named templates.
Run the Flask application: python app.py
Open the website at http://localhost:5000/.
Notes:
The frontend uses Bootstrap for styling and JQuery for handling form submission via AJAX.
The backend is a Flask application that saves user data and uploaded files in an SQLite database and a designated folder, respectively.
Make sure to handle file uploads securely. The example uses secure_filename from werkzeug.utils, which you should include in the imports.
Enhance the functionality with proper error handling, validation, and security measures, especially for handling file uploads.
Customize the frontend design further to align with the branding and features of the "GoFast" package.


"""

from flask import Flask, request, render_template
import os
import sqlite3

app = Flask(__name__)

# Directory to save uploaded files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize SQLite database
def init_db():
    with sqlite3.connect('gofast.db') as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                filename TEXT
            )
        """)
        conn.commit()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    name = request.form.get('name')
    email = request.form.get('email')
    file = request.files.get('file')
    filename = None

    if file and file.filename:
        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, filename))

    with sqlite3.connect('gofast.db') as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO users (name, email, filename) VALUES (?, ?, ?)
        """, (name, email, filename))
        conn.commit()

    return 'Data and file submitted successfully', 200

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
