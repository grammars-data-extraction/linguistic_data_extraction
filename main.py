from flask import Flask, render_template, request
from extractor import Extractor

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

ext = Extractor()

@app.route('/', methods=['GET', 'POST'])
def main_page():
    return render_template("main.html")


@app.route('/reranker.html', methods=['GET', 'POST'])
def reranker():
    return render_template("reranker.html")

@app.route('/bm25.html', methods=['GET', 'POST'])
def bm25():
    return render_template("bm25.html")

@app.route('/extracted.html', methods=['GET', 'POST'])
def extracted():
    return render_template("extracted.html")


@app.route('/reranker', methods=['GET', 'POST'])
def answer_reranker():

    feature = request.form["feature"]
    language = request.form["language"]
    res, indices, image_files, fname_indices = ext.extract(language, feature, method="Reranker")
    return render_template('extracted.html', res=res, indices=indices, image_files=image_files, fname_indices=fname_indices)

@app.route('/bm25', methods=['GET', 'POST'])
def answer_bm25():

    feature = request.form["feature"]
    language = request.form["language"]
    res, indices, image_files, fname_indices = ext.extract(language, feature, method="BM25")
    return render_template('extracted.html', res=res, indices=indices, image_files=image_files, fname_indices=fname_indices)

@app.errorhandler(500)
def internal_error(error):

    return render_template("error.html")

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

if __name__ == '__main__':
    app.run() 