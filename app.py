from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
from functools import wraps
import re

from keras.models import load_model
from PIL import Image
import numpy as np
import io
import base64

app = Flask(__name__)
model = load_model('model/model.h5')

app.secret_key = 'xyzsdfg'

app.config['MYSQL_HOST'] = '127.0.0.1'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'eyeu'

mysql = MySQL(app)

# Fungsi untuk memeriksa apakah pengguna sudah login


def check_logged_in():
    if 'loggedin' not in session:
        return False
    return True

# Dekorator untuk memeriksa login sebelum akses ke halaman


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not check_logged_in():
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


@app.route("/", methods=['GET', 'POST'])
@login_required
def home():
    id = session.get('id')
    message = ''
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    cursor.execute('SELECT * FROM user WHERE id = %s', (id,))
    user = cursor.fetchone()
    return render_template('home.html', user=user, message=message)



@app.route("/login", methods=['GET', 'POST'])
def login():
    mesage = ''
    print(request.form)
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute(
            'SELECT * FROM user WHERE email = % s AND password = % s', (email, password, ))
        user = cursor.fetchone()
        if user:
            session['loggedin'] = True
            session['id'] = user['id']
            session['name'] = user['name']
            session['email'] = user['email']
            session['password'] = user['password']
            return redirect(url_for('home'))
        else:
            mesage = 'Please enter correct email / password !'

    return render_template('login.html', mesage=mesage)


@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    session.pop('loggedin', None)
    session.pop('userid', None)
    session.pop('email', None)
    return redirect(url_for('login'))


@ app.route('/register', methods=['GET', 'POST'])
def register():
    mesage = ''
    if request.method == 'POST' and 'name' in request.form and 'password' in request.form and 'email' in request.form:
        name = request.form['name']
        password = request.form['password']
        email = request.form['email']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user WHERE email = % s', (email, ))
        account = cursor.fetchone()
        if account:
            mesage = 'Account already exists !'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            mesage = 'Invalid email address !'
        elif not name or not password or not email:
            mesage = 'Please fill out the form !'
        else:
            cursor.execute(
                'INSERT INTO user VALUES (NULL, % s, % s, % s)', (name, email, password, ))
            mysql.connection.commit()
            return redirect(url_for('login'))
    elif request.method == 'POST':
        mesage = 'Please fill out the form !'
    return render_template('register.html', mesage=mesage)


@app.route("/about", methods=['GET', 'POST'])
@login_required
def about():
    id = session.get('id')
    message = ''
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    cursor.execute('SELECT * FROM user WHERE id = %s', (id,))
    user = cursor.fetchone()
    return render_template('about.html', user=user, message=message)



@app.route('/profil', methods=['GET', 'POST'])
@login_required
def profil():
    id = session.get('id')
    message = ''
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    try:
        if request.method == 'POST' and 'name' in request.form and 'password' in request.form and 'email' in request.form:
            name = request.form['name']
            password = request.form['password']
            email = request.form['email']

            if not name or not password or not email:
                message = 'Mohon isi semua kolom!'
            else:
                cursor.execute(
                    'UPDATE user SET name = %s, email = %s, password = %s WHERE id = %s',
                    (name, email, password, id,)
                )
                mysql.connection.commit()
                return redirect(url_for('profil'))

        cursor.execute('SELECT * FROM user WHERE id = %s', (id,))
        user = cursor.fetchone()
        return render_template('profil.html', user=user, message=message)

    finally:
        cursor.close()


@app.route('/delete', methods=['GET', 'POST'])
@login_required
def delete():
    id = session.get('id')
    try:
        cursor = mysql.connection.cursor()
        cursor.execute('DELETE FROM user WHERE id = %s', (id,))
        mysql.connection.commit()
        return redirect(url_for('register'))
    finally:
        cursor.close()


@app.route("/konsultasi", methods=['GET', 'POST'])
@login_required
def konsultasi():

    id = session.get('id')
    message = ''
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    cursor.execute('SELECT * FROM user WHERE id = %s', (id,))
    user = cursor.fetchone()
    return render_template('konsultasi.html', user=user, message=message)


@app.route("/blog")
def blog():
    return render_template("blog1.html")


@app.route("/tester")
def tester():
    return render_template("tester.html")


@app.route("/produk", methods=['GET', 'POST'])
@login_required
def produk():
    id = session.get('id')
    message = ''
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    cursor.execute('SELECT * FROM user WHERE id = %s', (id,))
    user = cursor.fetchone()
    return render_template('produk.html', user=user, message=message)


@app.route("/chat", methods=['GET', 'POST'])
@login_required
def chat():
    id = session.get('id')
    message = ''
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    cursor.execute('SELECT * FROM user WHERE id = %s', (id,))
    user = cursor.fetchone()
    return render_template('chat.html', user=user, message=message)


@app.route("/chat/dr-bima-arya-fatah-sp-m", methods=['GET', 'POST'])
@login_required
def chat_dr_bima():
    id = session.get('id')
    message = ''
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    cursor.execute('SELECT * FROM user WHERE id = %s', (id,))
    user = cursor.fetchone()
    return render_template('dr.bima.html', user=user, message=message)


@app.route('/deteksi', methods=['GET', 'POST'])
@login_required
def deteksi():
    if request.method == 'POST':
        # Cek apakah file telah dipilih
        if 'file' not in request.files:
            return render_template('deteksi.html', message='File belum dipilih')

        file = request.files['file']

        # Cek apakah file kosong
        if file.filename == '':
            return render_template('deteksi.html', message='File belum dipilih')

        # Simpan file ke direktori tertentu (opsional)
        file_path = 'histori/' + file.filename
        file.save(file_path)

        # Proses file menggunakan model
        img = Image.open(file_path)
        img = np.array(img.resize((94, 55)))
        img = np.expand_dims(img, axis=0)
        pred = model.predict(img)

        # Hapus file yang diunggah setelah diproses (opsional)
        # import os
        # os.remove(file_path)

        # Tampilkan hasil prediksi di halaman web
        # actual_class = 'normal'  # sesuaikan dengan kelas yang sesuai dengan gambar
        predicted_class = 'Normal' if pred[0] > 0.5 else 'Katarak'

        # Konversi gambar ke format yang dapat ditampilkan di HTML
        img_str = image_to_base64(img)

        return render_template('deteksi.html', message=f'Prediksi: {predicted_class}',
                               predicted_class=predicted_class, img_str=img_str)

    return render_template('deteksi.html')


def image_to_base64(image):
    # Konversi gambar menjadi format base64 agar bisa ditampilkan di HTML
    img = Image.fromarray(image[0])
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = "data:image/png;base64," + \
        base64.b64encode(buffer.getvalue()).decode()
    return img_str


if __name__ == '__main__':
    app.run(debug=True)
