from flask import Flask, request, jsonify, render_template
from detectors import analyze

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB

@app.errorhandler(Exception)
def on_error(e):
    return jsonify(error=f"Server error: {e}"), 500

@app.get("/")
def home():
    return render_template("index.html")

@app.post("/api/check")
def check():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="Koi file nahi mili"), 400
    try:
        res = analyze(f.filename, f.read())
    except ValueError as e:
        return jsonify(error=str(e)), 415
    except Exception as e:
        return jsonify(error=f"File process nahi ho payi: {e}"), 500
    s = res["score"]
    res["verdict"] = ("Nateeja tay nahi ho saka" if s is None else
                      "AI hone ke majboot signal mile" if s >= 75 else
                      "Kuch signal mile, pakka nahi" if s >= 40 else
                      "Koi khas AI signal nahi mila")
    res["filename"] = f.filename
    return jsonify(res)

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False, threaded=True, port=5000)
