from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
from functools import wraps
import re


from keras.models import load_model
from PIL import Image
import numpy as np
import io
import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import base64
import json
import random
intents = json.loads(open('chatbot/intents.json').read())
words = pickle.load(open('chatbot/words2.pkl','rb'))
classes = pickle.load(open('chatbot/classes2.pkl','rb'))

app = Flask(__name__)
model = load_model('model/model.h5')
model = load_model('chatbot/chatbot_model2.h5')

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


@app.route("/riwayat-deteksi", methods=['GET', 'POST'])
@login_required
def history():
    id = session.get('id')
    message = ''
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    cursor.execute('SELECT * FROM user WHERE id = %s', (id,))
    user = cursor.fetchone()
    return render_template('history.html', user=user, message=message)

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

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)


def image_to_base64(image):
    # Konversi gambar menjadi format base64 agar bisa ditampilkan di HTML
    img = Image.fromarray(image[0])
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = "data:image/png;base64," + \
        base64.b64encode(buffer.getvalue()).decode()
    return img_str

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


@app.route("/feedback")
def feedback():
    return render_template("feedback.html")

if __name__ == '__main__':
    app.run(debug=True)
