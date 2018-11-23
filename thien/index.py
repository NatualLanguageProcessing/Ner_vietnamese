from flask import Flask, jsonify, render_template, request
import model
app = Flask(__name__)

crf = model.CRF()

@app.route('/get_ner', methods=['GET', 'POST'])
def get_ner():
    if request.method == "POST":
        paragraph = request.form['text']
        print(paragraph)
        entities, tokenizes = crf.get_entity(paragraph)
        return jsonify(dict({'entities': entities, 'tokenizes': tokenizes}))

@app.route('/')
def index():
    return render_template('index.html')


app.run(host='0.0.0.0', port=5000, debug=True)