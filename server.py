from flask import Flask, render_template, send_from_directory, url_for, request, redirect, abort
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from sqlalchemy import ForeignKey
from wtforms import SubmitField
from flask_sqlalchemy import SQLAlchemy
import tensorflow as tf
import cv2
import math
import numpy as np

global model
model = tf.keras.models.load_model('/Users/mm22/Documents/Development/twentypercentplus/models/20dogs5cats/model_one.h5')

categories = ['Abyssinian', 'Afghan_hound', 'American_Shorthair', 'Australian_terrier', 'basset', 'Beagle', 
              'Bengal', 'bloodhound', 'Border_collie', 'borzoi', 'Chihuahua', 'cocker_spaniel', 'dingo', 
              'Doberman', 'French_bulldog', 'German_shepherd', 'Golden_retriever', 'Great_Dane', 'Maltese_dog', 
              'Persian', 'pug', 'Samoyed', 'Shih-Tzu', 'Siamese', 'toy_poodle']

class TensorVector(object):

    def __init__(self, FileName=None):
        self.FileName = FileName

    def process(self):

        img = tf.io.read_file(self.FileName)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize_with_pad(img, 299, 299)
        img = tf.image.convert_image_dtype(img,tf.float32)[tf.newaxis, ...]
        features = model(img)
        feature_set = np.squeeze(features)
        return list(feature_set)
    
def cosineSim(a1,a2):
    sum = 0
    suma1 = 0
    sumb1 = 0
    for i,j in zip(a1, a2):
        suma1 += i * i
        sumb1 += j*j
        sum += i*j
    cosine_sim = sum / ((math.sqrt(suma1))*(math.sqrt(sumb1)))
    return cosine_sim

def ed(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must be of equal length.")
    squared_distance = 0
    for i in range(len(vector1)):
        squared_distance += (vector1[i] - vector2[i]) ** 2
    return math.sqrt(squared_distance)

def prepare(filepath):
    IMG_SIZE = 299
    img_array = cv2.imread(filepath)[...,::-1]
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)


app = Flask(__name__)
app.config['SECRET_KEY'] = 'abc'
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///pets.db'

db = SQLAlchemy(app)

class Pets(db.Model):
    pet_id = db.Column(db.Integer, primary_key=True)
    pet_name = db.Column(db.String(50), nullable=False)
    species = db.Column(db.String(50), nullable=False)
    breed = db.Column(db.String(50), nullable=False)
    birthdate = db.Column(db.String(50), nullable=False)
    weight = db.Column(db.String(50), nullable=False)
    reproductive_status = db.Column(db.String(50), nullable=False)

    def __repr__(self):
        #return f'{self.pet_name} : {self.pet_id}'
        return str(self.pet_id)

class PetImages(db.Model):
    photo_id = db.Column(db.Integer, primary_key=True)
    pet_id = db.Column(db.Integer, ForeignKey(Pets.pet_id))
    pet_name = db.Column(db.String(50), ForeignKey(Pets.pet_name))
    img_path = db.Column(db.String(100), nullable=False)

    def __repr__(self):
        return '<Name %r>' % self.pet_id


photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)

class UploadForm(FlaskForm):
    photo = FileField(
        validators=[
            FileAllowed(photos, 'Only Images are allowed'),
            FileRequired('File field should not be empty')
        ]
    )
    submit = SubmitField('Upload')

# main landing page where the pet recognition feature is
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = url_for('get_file', filename=filename)
        prediction = model.predict([prepare(f'uploads/{filename}')])
        prediction_list = list(prediction[0])
        prediction_index = prediction_list.index(max(prediction_list))
        accuracy = round((max(prediction_list) * 100), 2)
        if accuracy < 50.0:
            result = 'Not a Dog or Cat'
        else:
            result = categories[prediction_index]
    else:
        file_url = None
        accuracy = 0
        result = None
    return render_template('index.html', form=form, file_url=file_url, accuracy=accuracy, result = result)

def similarity(filename):
    cosineSim_values = []
    euclidDist_values = []
    petimages = [ p.img_path for p in PetImages.query.all()]

    h1 = TensorVector(filename)
    v1 = h1.process()

    for i in range(len(petimages)):
        h2 = TensorVector(petimages[i])
        v2 = h2.process()
        cosineSim_values.append(cosineSim(v1, v2))
        euclidDist_values.append(ed(v1, v2))

    cosinemaxIndex = cosineSim_values.index(max(cosineSim_values))
    euclidDistminIndex = euclidDist_values.index(min(euclidDist_values))
    print(cosinemaxIndex, euclidDistminIndex)
    if (cosineSim_values[cosinemaxIndex] > .800) and (cosinemaxIndex == euclidDistminIndex or cosinemaxIndex + 1 == euclidDistminIndex or cosinemaxIndex - 1 == euclidDistminIndex):
        animal_column = PetImages.query.filter_by(photo_id = cosinemaxIndex + 1).first()
        return animal_column.pet_id
    else:
        return 'No match'

# pet identification page
@app.route('/pet/finder', methods=['GET', 'POST'])
def find_pet():
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = url_for('get_file', filename=filename)
        output = similarity(f'uploads/{filename}')

        if output != 'No match':
            matchimages = PetImages.query.filter_by(pet_id=output).all()
            match_urls = ['/' + p.img_path for p in matchimages]
            pet_name = (Pets.query.filter_by(pet_id=output).first()).pet_name
        else:
            match_urls = []
            pet_name = 'This animal is not in our system'
    else:
        file_url = None
        match_urls = []
        pet_name = ''
    return render_template('finder.html', form=form, file_url=file_url, match_urls=match_urls, pet_name = pet_name)


@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], filename)

#CRUD database where pets are created,retrieved,updated and deleted

@app.route('/pet/create', methods=['GET', 'POST'])
def create():
    if request.method == 'GET':
        return render_template('createpet.html')
    
    if request.method == 'POST':
        pet_name = request.form['pet_name']
        species = request.form['species']
        breed = request.form['breed']
        birthdate = request.form['birthdate']
        weight = request.form['weight']
        reproductive_status = request.form['reproductive_status']
        pet = Pets(pet_name=pet_name, species=species,breed=breed,birthdate=birthdate
                            ,weight=weight,reproductive_status=reproductive_status)
        db.session.add(pet)
        db.session.commit()
        return redirect('/pet')
    
@app.route('/pet/<int:pet_id>/uploadphotos', methods=['GET', 'POST'])
def upload_pet_photos(pet_id):
    form = UploadForm()
    
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = url_for('get_file', filename=filename)
        img_path = f'uploads/{filename}'

        petimage = PetImages(pet_id = pet_id,img_path = img_path)
        db.session.add(petimage)
        db.session.commit()
    else:
        file_url=None
    return render_template('uploadpetphotos.html', form=form, file_url=file_url)


@app.route('/pet')
def RetrievePets():
    pets = Pets.query.all()
    return render_template('petlist.html', pets=pets)

@app.route('/pet/<int:pet_id>')
def RetrievePet(pet_id):
    pet = Pets.query.filter_by(pet_id=pet_id).first()
    petimage = PetImages.query.filter_by(pet_id=pet_id).all()
    image_urls = ['/' + p.img_path for p in petimage]
    if pet:
        return render_template('pet.html', pet = pet, petimage=petimage, image_urls=image_urls)
    return f'Pet with id={pet_id} is not in our system.'

@app.route('/pet/<int:pet_id>/update', methods=['GET', 'POST'])
def update(pet_id):

    pet = Pets.query.filter_by(pet_id=pet_id).first()

    if request.method == 'POST':
        if pet:
            db.session.delete(pet)
            db.session.commit()
            pet_name = request.form['pet_name']
            species = request.form['species']
            breed = request.form['breed']
            birthdate = request.form['birthdate']
            weight = request.form['weight']
            reproductive_status = request.form['reproductive_status']
            pet = Pets(pet_name=pet_name, species=species,breed=breed,birthdate=birthdate
                       ,weight=weight,reproductive_status=reproductive_status)
            db.session.add(pet)
            db.session.commit()

            return redirect(f'/pet/{pet_id}')
        return f'Pet with id = {pet_id} is not in our system.'
    return render_template('petupdate.html', pet=pet)

@app.route('/pet/<int:pet_id>/delete')
def delete(pet_id):
    pet = Pets.query.filter_by(pet_id=pet_id).first()
    if request.method == 'POST':
        if pet:
            db.session.delete(pet)
            db.session.commit()
            return redirect('/pet')
        abort(404)

    return render_template('deletepet.html')

if __name__ == '__main__':
    app.run(debug=True)