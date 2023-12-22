"""
Creating a solution that involves both Python and JavaScript to collect data from users through a website and save it in an SQLite database involves several steps. Here's a high-level overview and a basic implementation:

High-Level Overview:
Frontend (JavaScript/HTML): Create a simple form to collect user data.
Backend (Python/Flask): Set up a Flask server to handle POST requests from the frontend form.
Database (SQLite): Use SQLite to store the data received from the user.
Implementation:
Frontend: HTML/JavaScript Form
Create an HTML file (index.html) with a form:


This Python code uses Flask to serve the HTML form and handle form submissions. When the form 
is submitted, it saves the data into an SQLite database.

Running the Application:
Install Flask: pip install flask
Place the index.html file in a folder named templates in the same directory as your app.py.
Run the Flask app: python app.py
Open your web browser and go to http://localhost:5000/.
Now, when you fill out the form on the webpage and submit it, the data will be sent to the Flask server, 
which then saves it to the SQLite database.

This basic example is just a starting point. In a real-world application, you should add error handling, 
input validation, security measures against SQL injection, and use AJAX for form submission for a better 
user experience.
 Also, consider using SQLAlchemy for more complex database operations.
"""



from flask import Flask, request, render_template
import sqlite3

app = Flask(__name__)

def init_db():
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    name = request.form['name']
    email = request.form['email']
    conn = sqlite3.connect('user_data.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO users (name, email) VALUES (?, ?)', (name, email))
    conn.commit()
    conn.close()
    return 'Data saved successfully', 200

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
