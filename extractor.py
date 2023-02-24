import os
import re
import pandas as pd
from langdetect import detect
import textract
import json
from rank_bm25 import BM25Okapi
from urllib.request import urlopen
from bs4 import BeautifulSoup
import wikipedia
import spacy
import pdftotext
import PyPDF2
import re
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
import scipy
import numpy as np


class Extractor:
    def __init__(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.df = pd.read_excel("linguistic_data_extraction/grammars_database.xlsx")
        self.lang_list = ["ca", "zh", "en", "fr", "de", "it", "pt", "ru", "es", "nl"]
        self.models = dict()
        self.stopwords = dict()
        self.embedder = SentenceTransformer('bert-base-multilingual-cased')

        modelPath = "bert-base-multilingual-cased"

        self.embedder.save(modelPath)
        self.embedder = SentenceTransformer(modelPath)

        with open("linguistic_data_extraction/data/language_files.json", "r") as file:
            self.language_files = json.load(file)

        for lang in self.lang_list:
            exec("from spacy.lang.%s.stop_words import STOP_WORDS" % (lang))
            exec("self.stopwords[lang] = spacy.lang.%s.stop_words.STOP_WORDS" % (lang))
            if lang in ("en", "zh"):
                model_name = lang + "_core_web_sm"
            else:
                model_name = lang + "_core_news_sm"

            self.models[lang] = spacy.load(model_name, disable=['parser', 'ner'])

    def get_lang(self, filename):
        text = textract.process(filename).decode("utf-8") 
        lang = detect(text)
        return lang

    def first_letter(self, s):
        m = re.search(r'[a-z]', s, re.I)
        if m is not None:
            return s[m.start()]
        return "A"

    def end_of_sentence(self, text):
        text = text.strip("\n ")
        stop = ('...', '.', '?', '!', '!!!', '…')
        for item in stop:
            if text.endswith(item):
                return True
        return False

    def digits(self, text):
        num = 0
        for i in text:
            if i.isdigit():
                num += 1
        return num

    def get_term(self, term, language):
        if language == "en":
            return term
        # get languages
        soup = BeautifulSoup(urlopen('http://en.wikipedia.org/wiki/' + (term[:1].upper() + term[1:]).replace(' ', '_')), features="lxml")
        interwikihead = soup.find('li', class_=('interlanguage-link interwiki-' + language + ' mw-list-item'))

        try:
            title = re.split(' \(| –', interwikihead.a.get('title'))[0]
            return title.lower()
        except:
            return None 

    def get_description(self, term, language):
        soup = BeautifulSoup(urlopen('http://en.wikipedia.org/wiki/' + (term[:1].upper() + term[1:]).replace(' ', '_')), features="lxml")
        interwikihead = soup.find('li', class_=('interlanguage-link interwiki-' + language + ' mw-list-item'))

        try:
            if language == "en":
                title = term[:1].upper() + term[1:]
            else:
                title = interwikihead.a.get('title').split(u' – ')[0]
            wikipedia.set_lang(language)
            page = wikipedia.page(title, auto_suggest=False)
            return page.summary
        except:
            return None 

    def get_new_paragraphs(self, paragraphs, page_numbers):
        new_paragraphs = [paragraphs[0]]
        new_page_numbers = dict()
        new_page_numbers[paragraphs[0]] = [page_numbers[0]]
        for i in range(1, len(paragraphs)):
            paragraph = paragraphs[i].strip(" \n")
            if len(paragraph) > 0:
                if (not self.end_of_sentence(new_paragraphs[-1])) | self.first_letter(paragraphs[i]).islower():
                    index = new_page_numbers[new_paragraphs[-1]]
                    del new_page_numbers[new_paragraphs[-1]]
                    new_paragraphs[-1] += paragraphs[i]
                    if index[-1] != page_numbers[i]:
                        index.append(page_numbers[i])
                    new_page_numbers[new_paragraphs[-1]] = index
                else:
                    new_paragraphs.append(paragraphs[i])
                    new_page_numbers[paragraphs[i]] = [page_numbers[i]]
        return new_paragraphs, new_page_numbers

    def make_dir(self, fname):

        directory = os.path.dirname(fname)
        if not os.path.exists(directory):
            os.system("mkdir -p \"{directory}\"")
        os.system("touch \"{os.path.basename(new_fname)}\"")


    def extract_image(self, pdf_file, i, nums):

        image_names = []
        output_fname = "linguistic_data_extraction/static/image/output.pdf"
        pdf_writer = PyPDF2.PdfWriter()

        for num in nums:
            page = pdf_file.pages[num]
            pdf_writer.add_page(page)

        with open(output_fname, 'wb') as out:
            pdf_writer.write(out)

        images = convert_from_path(output_fname)
        for image in images:
            image_name = "output" + str(i) + ".jpg"
            image_names.append(image_name)
            image.save("linguistic_data_extraction/static/image/" + image_name, 'JPEG')
            i += 1
        return image_names, i

    def rerank(self, query, entries, embeddings_dict):

        corpus_embeddings = self.embedder.encode(entries)

        query_embedding = self.embedder.encode(query)

        # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
        closest_n = 5
        distances = scipy.spatial.distance.cdist(query_embedding, corpus_embeddings, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        ans = []

        for idx, distance in results[0:closest_n]:
            ans.append(entries[idx])

        return ans


    def extract(self, lang_about, query, method, description=True):

        #try:

            ans = []
            image_files = []
            fnames = []
            i = 0
            filenames = self.language_files[lang_about]
            for item in filenames:

                fname = item[0]
                short_name = fname

                lang = item[1]
                absolute_path = "linguistic_data_extraction/data/"
                fname = absolute_path + fname
                fname_par = (absolute_path + "Grammars_Paragraphs/" + os.path.basename(fname)).replace("pdf", "json")
                fname_lem = (absolute_path + "Grammars_Lemmas/" + os.path.basename(fname)).replace("pdf", "json")
                fname_num = (absolute_path + "Grammars_Page_Numbers/" + os.path.basename(fname)).replace("pdf", "json")


                nlp = self.models[lang]
                stopwords_lang = self.stopwords[lang]

                if not os.path.exists(fname_par):
                    
                    command = "rclone copy \"gdrive:" + short_name + "\" \"linguistic_data_extraction/data/" + os.path.dirname(short_name) + "\" --no-traverse --drive-chunk-size 32M -P"
                    print(command)
                    os.system(command)

                    with open(fname, 'rb') as f:
                        pdf = pdftotext.PDF(f)
                    page_numbers = []
                    paragraphs = []
                    lemmatized_paragraphs = []
                    for j in range (len(pdf)):
                        addition = re.split('  ', pdf[j])
                        paragraphs.extend(addition)
                        for paragraph in addition:
                            page_numbers.append(j)
                    new_paragraphs, new_page_numbers = self.get_new_paragraphs(paragraphs, page_numbers)

                    for paragraph in new_paragraphs:
                        lemmatized_paragraph = []
                        doc = nlp(paragraph.lower())
                        for token in doc:
                            if token.lemma_ not in stopwords_lang and token.is_alpha:
                                lemmatized_paragraph.append(token.lemma_)
                        lemmatized_paragraphs.append(lemmatized_paragraph)

                    self.make_dir(fname_lem)
                    self.make_dir(fname_par)
                    self.make_dir(fname_num)

                    with open(fname_lem, 'w') as outfile:
                        lem = json.dumps(lemmatized_paragraphs)
                        outfile.write(lem)
                    with open(fname_par, 'w') as outfile:
                        par = json.dumps(new_paragraphs)
                        outfile.write(par)
                    with open(fname_num, 'w') as outfile:
                        num = json.dumps(new_page_numbers)
                        outfile.write(num)

                else:
                    with open(fname_par, 'r') as file:
                        new_paragraphs = json.load(file)
                    with open(fname_lem, 'r') as file:
                        lemmatized_paragraphs = json.load(file)
                    with open(fname_num, 'r') as file:
                        new_page_numbers = json.load(file)

                pdf_file = PyPDF2.PdfReader(fname)
                bm25 = BM25Okapi(lemmatized_paragraphs)

                if description:
                    term = self.get_term(query, lang)
                    fname_desc = "linguistic_data_extraction/data/Grammars_Summaries/" + query + "_" + lang + ".json"
                    fname_desc_lem = "linguistic_data_extraction/data/Grammars_Summaries/" + query + "_" + lang + "_lemmatized.json"

                    if not os.path.exists(fname_desc):
                        desc = self.get_description(query, lang)
                        desc_nlp = nlp(self.get_description(query, lang))
                        lemmatized_desc = []

                        for token in desc_nlp:
                            if token.lemma_ not in stopwords_lang and token.is_alpha:
                                lemmatized_desc.append(token.lemma_)

                        self.make_dir(fname_desc)
                        self.make_dir(fname_desc_lem)

                        with open(fname_desc, 'w') as outfile:
                            file_desc = json.dumps(desc)
                            outfile.write(file_desc)
                        with open(fname_desc_lem, 'w') as outfile:
                            file_desc_lem = json.dumps(lemmatized_desc)
                            outfile.write(file_desc_lem)

                    else:
                        with open(fname_desc, 'r') as file:
                            desc = json.load(file)
                        with open(fname_desc_lem, 'r') as file:
                            lemmatized_desc = json.load(file)

                    query_translated = lemmatized_desc

                else:
                    term = nlp(self.get_term(query, lang))
                    query_translated = []

                    for token in term:
                        if token.lemma_ not in stopwords_lang and token.is_alpha:
                            query_translated.append(token.lemma_)

                top_n = bm25.get_top_n(query_translated, new_paragraphs, n=5)

                if method == "BM25":
                    ans += top_n
                    for item in top_n:

                        num = new_page_numbers[item]
                        image_names, i = self.extract_image(pdf_file, i, num)
                        image_files.append(image_names)

                else:
                    reranked = self.rerank(query_translated, top_n, dict())
                    ans += reranked

                    for item in reranked:

                        num = new_page_numbers[item]
                        image_names, i = self.extract_image(pdf_file, i, num)
                        image_files.append(image_names)

                fnames.append(short_name)

            fname_indices = []

            for fname in fnames:
                for k in range(5):
                    fname_indices.append([fname, k + 1]) 
            indices = list(range(len(ans)))
            
            return ans, indices, image_files, fname_indices

        #except:
            #pass