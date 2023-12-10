from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from model.utils import linear_transform, nonlinear_transform, hist_equalization_transform, hist_specification_transform

UPLOAD_FOLDER = "static/images"
UPLOAD_GRAY_FILE = "static/images/gray"
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config["UPLOAD_GRAY_FILE"] =UPLOAD_GRAY_FILE

@app.route('/')
def home():
    return render_template("upload.html")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect("/")
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    else:
        return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # img_dir = send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    img_dir = "images" + '/' + filename
    return render_template("uploaded.html", img_dir=img_dir, img_name=filename)


@app.route("/color_transformation/brightness/<filename>", methods=['GET', 'POST'])
def brightness_trans(filename):
    img = cv2.imread(UPLOAD_FOLDER + "/" + filename)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    BRIGHTNESS_FOLDER = "images/brightness/" + filename
    if request.method == 'POST':
        bias = float(request.form["brightness_range"])
        trans_img = linear_transform(img, bias=bias, weight=1)
        cv2.imwrite("static/images/brightness/" + filename, trans_img)
        return render_template("brightness.html", img_dir=BRIGHTNESS_FOLDER, filename=filename)
    else:
        bias = 0
        trans_img = img
        cv2.imwrite("static/images/brightness/" + filename, trans_img)
        return render_template("brightness.html", img_dir=BRIGHTNESS_FOLDER, filename=filename)


@app.route("/color_transformation/contrast/<filename>", methods=['GET', 'POST'])
def contrast_trans(filename):
    img = cv2.imread(UPLOAD_FOLDER + "/" + filename)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    CONTRAST_FOLDER = "images/contrast/" + filename
    weight = 1
    if request.method == 'POST':
        weight = float(request.form["contrast_range"])
        trans_img = linear_transform(img, weight=weight)
        cv2.imwrite("static/images/contrast/" + filename, trans_img)
        return render_template("contrast.html", img_dir=CONTRAST_FOLDER, filename=filename)
    else:
        weight = 1
        trans_img = linear_transform(img, weight=weight)
        cv2.imwrite("static/images/contrast/" + filename, trans_img)
        return render_template("contrast.html", img_dir=CONTRAST_FOLDER, filename=filename)


@app.route("/color_transformation/brightness_and_contrast/<filename>", methods=['GET', 'POST'])
def brightness_and_contrast_trans(filename):
    img = cv2.imread(UPLOAD_FOLDER + "/" + filename)
    BRIGHTNESS_AND_CONTRAST_FOLDER = "images/brightness_and_contrast/" + filename
    weight = 1
    bias = 0

    if request.method == 'POST':
        weight = float(request.form["contrast_range"])
        bias = float(request.form["brightness_range"])
        trans_img = linear_transform(img, bias=bias, weight=weight)
        cv2.imwrite("static/images/brightness_and_contrast/" + filename, trans_img)
        return render_template("brightness_contrast.html", img_dir=BRIGHTNESS_AND_CONTRAST_FOLDER, filename=filename)
    else:
        weight = 1
        bias = 0
        trans_img = linear_transform(img, bias=bias, weight=weight)
        cv2.imwrite("static/images/brightness_and_contrast/" + filename, trans_img)
        return render_template("brightness_contrast.html", img_dir=BRIGHTNESS_AND_CONTRAST_FOLDER, filename=filename)


@app.route("/color_transformation/logarithm/<filename>", methods = ["GET", "POST"])
def logarithm_trans(filename):
    img = cv2.imread(UPLOAD_FOLDER + "/" + filename)
    LOGARITHM_FOLDER = "images/logarithm/" + filename
    if request.method == 'POST':
        const = int(request.form["const_range"])
        trans_img = nonlinear_transform(img, key="log", const=const)
        cv2.imwrite("static/images/logarithm/" + filename, trans_img)
        return render_template("logarithm.html", img_dir=LOGARITHM_FOLDER, filename=filename)
    else:
        const = 10
        trans_img = nonlinear_transform(img, key="log", const=const)
        cv2.imwrite("static/images/logarithm/" + filename, trans_img)
        return render_template("logarithm.html", img_dir=LOGARITHM_FOLDER, filename=filename)

@app.route("/color_transformation/exponential/<filename>", methods = ["GET"])
def exponential_trans(filename):
    img = cv2.imread(UPLOAD_FOLDER + "/" + filename)
    EXPONENTIAL_FOLDER = "images/exponential/" + filename
    trans_img = nonlinear_transform(img, key="exp")
    cv2.imwrite("static/images/exponential/" + filename, trans_img)
    return render_template("exponential.html", img_dir=EXPONENTIAL_FOLDER, filename=filename)


@app.route("/color_transformation/histogram_equalization/<filename>", methods = ["GET"])
def his_equal_trans(filename):
    img = cv2.imread(UPLOAD_FOLDER + "/" + filename)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("static/images/gray/" + filename, gray_image)
    HIS_EQUAL_FOLDER = "images/his_equal/" + filename
    trans_img = hist_equalization_transform(gray_image)
    cv2.imwrite("static/images/his_equal/" + filename, trans_img)
    return render_template("histogram_equal.html", gray_img_dir="images/gray/" + filename, ori_img_dir="images/" + filename, img_dir=HIS_EQUAL_FOLDER, filename=filename)


@app.route("/color_transformation/histogram_specification/<filename>/<gray_filename>", methods = ["GET"])
def his_spec_trans(filename, gray_filename):
    HIS_SPEC_FOLDER = "images/his_spec/" + filename

    img = cv2.imread(UPLOAD_FOLDER + "/" + filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("static/images/" + filename, img)
    gray_img = cv2.imread(UPLOAD_GRAY_FILE + "/" + gray_filename)
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)

    trans_img = hist_specification_transform(img, gray_img)
    cv2.imwrite("static/images/his_spec/" + filename, trans_img)
    cv2.imwrite("static/images/gray/" + gray_filename, gray_img)
    return render_template("histogram_spec.html", gray_img_dir="images/gray/" + gray_filename, ori_img_dir="images/" + filename, img_dir=HIS_SPEC_FOLDER, filename=filename)


@app.route('/upload_gray_file/<filename>', methods=['GET', 'POST'])
def upload_gray_file(filename):
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect("/upload_gray_file/<filename>")
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            gray_filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_GRAY_FILE'], gray_filename))
            return redirect(url_for('his_spec_trans',
                                    filename=filename, gray_filename=gray_filename))
    else:
        return render_template('upload_gray_img.html')

if __name__ == '__main__':
    app.run(debug=True)
