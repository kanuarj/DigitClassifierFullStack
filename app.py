from flask import Flask, render_template, request
from prediction import prediction_function
app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])

def basic():
    if request.method == 'POST':
        input_link = request.form['input_link']
        y_pred = prediction_function(input_link)
        return render_template('index.html', output = y_pred)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug = True)