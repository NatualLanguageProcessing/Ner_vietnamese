from flask import Flask, jsonify, render_template, request, url_for
import model
app = Flask(__name__)

crf = model.CRF_Ner()
lstm_crf = model.LSTM_CRF(load_model=True)

@app.route('/get_ner', methods=['GET', 'POST'])
def get_ner():
    if request.method == "POST":
        paragraph = request.form['text']
        
        entities, tokenizes = crf.get_entity(paragraph)
        # entities, tokenizes = lstm_crf.get_ner(paragraph)
        print(entities)
        return jsonify(dict({'entities': entities, 'tokenizes': tokenizes}))

@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
	app.run(debug=True)