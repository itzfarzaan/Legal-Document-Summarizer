from flask import Flask, render_template, request, jsonify
from langchain.document_loaders import PyPDFLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from googletrans import Translator

app = Flask(__name__)

# Load the tokenizer and model when the app is opened
tokenizer = AutoTokenizer.from_pretrained("nsi319/legal-pegasus", force_download=True)
model = AutoModelForSeq2SeqLM.from_pretrained("nsi319/legal-pegasus", force_download=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    uploaded_file = request.files['file']
    if uploaded_file:
        # Load the PDF document
        file_path = f"/tmp/{uploaded_file.filename}"
        uploaded_file.save(file_path)
        pdf_loader = PyPDFLoader(file_path)
        doc = pdf_loader.load()[0]

        text = doc.page_content
        input_tokenized = tokenizer.encode(text, return_tensors='pt', max_length=1024, truncation=True)
        summary_ids = model.generate(input_tokenized,
                                     num_beams=9,
                                     no_repeat_ngram_size=3,
                                     length_penalty=2.0,
                                     min_length=150,
                                     max_length=250,
                                     early_stopping=True)

        summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]
        return jsonify({'summary': summary, 'text': text})

@app.route('/translate', methods=['POST'])
def translate():
    summary = request.form['summary']
    target_language = request.form['language']
    translator = Translator()
    translated_summary = translator.translate(summary, dest=target_language).text
    return jsonify({'translated_summary': translated_summary})

if __name__ == '__main__':
    app.run(debug=True)