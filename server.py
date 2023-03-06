from flask import Flask, render_template, send_from_directory, url_for
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
import tensorflow as tf
import cv2

model = tf.keras.models.load_model('/Users/mm22/Documents/Development/twentypercentplus/models/20dogs5cats/model_one.h5')

categories = ['Abyssinian', 'Afghan_hound', 'American_Shorthair', 'Australian_terrier', 'basset', 'Beagle', 
              'Bengal', 'bloodhound', 'Border_collie', 'borzoi', 'Chihuahua', 'cocker_spaniel', 'dingo', 
              'Doberman', 'French_bulldog', 'German_shepherd', 'Golden_retriever', 'Great_Dane', 'Maltese_dog', 
              'Persian', 'pug', 'Samoyed', 'Shih-Tzu', 'Siamese', 'toy_poodle']

def prepare(filepath):
    IMG_SIZE = 299
    #img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img_array = cv2.imread(filepath)[...,::-1]
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)


app = Flask(__name__)
app.config['SECRET_KEY'] = 'abc'
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'

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
        result = categories[prediction_index]
    else:
        file_url = None
    return render_template('index.html', form=form, file_url=file_url, accuracy=accuracy, result = result)


@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], filename)


if __name__ == '__main__':
    app.run(debug=True)