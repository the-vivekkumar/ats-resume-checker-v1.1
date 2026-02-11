import pdfplumber
import docx
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO

app = Flask(__name__)


# -------------------------
# Extract PDF directly
# -------------------------
def extract_pdf(file_stream):
    text = ""
    with pdfplumber.open(file_stream) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


# -------------------------
# Extract DOCX directly
# -------------------------
def extract_docx(file_stream):
    doc = docx.Document(file_stream)
    return "\n".join([p.text for p in doc.paragraphs])


# -------------------------
# Calculate ATS Score
# -------------------------
def calculate_score(resume, job):

    if not resume.strip() or not job.strip():
        return 0

    docs = [resume, job]

    vectorizer = CountVectorizer().fit_transform(docs)
    vectors = vectorizer.toarray()

    similarity = cosine_similarity(vectors)

    return round(similarity[0][1] * 100, 2)


# -------------------------
# Route
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():

    score = None
    error = None

    if request.method == "POST":

        try:
            file = request.files["resume"]
            job_desc = request.form["jobdesc"]

            if not file:
                error = "Upload a file"
                return render_template("index.html", score=score, error=error)

            filename = file.filename.lower()

            # 🔥 NO SAVING ANYWHERE
            file_bytes = BytesIO(file.read())

            if filename.endswith(".pdf"):
                resume_text = extract_pdf(file_bytes)

            elif filename.endswith(".docx"):
                resume_text = extract_docx(file_bytes)

            else:
                error = "Only PDF or DOCX allowed"
                return render_template("index.html", score=score, error=error)

            score = calculate_score(resume_text, job_desc)

        except Exception as e:
            error = str(e)

    return render_template("index.html", score=score, error=error)


if __name__ == "__main__":
    app.run(debug=True)
