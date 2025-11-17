import os
import shutil
import zipfile
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session, make_response
from werkzeug.utils import secure_filename
from flask_cors import CORS
from bashfile.create_VI import create_VI
# --- Your function ---
from bashfile.bashfile import main_function  

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "super_secret_key_change_me")
CORS(app, origins=os.environ.get("FRONTEND_URL", "*"))

# --- Directories ---
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
OUTPUT_FOLDER = BASE_DIR / "output"
TEMP_EXTRACT = BASE_DIR / "temp"

for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, TEMP_EXTRACT]:
    folder.mkdir(exist_ok=True)

# ----------------------------
# 🔐 Middleware for token check
# ----------------------------
def is_logged_in():
    token = request.cookies.get("firebase_token")
    return bool(token)  # token presence indicates logged in

@app.before_request
def protect_routes():
    # list of endpoints to protect
    protected = ["index", "download_file"]
    if request.endpoint in protected and not is_logged_in():
        return redirect(url_for("login"))

# ----------------------------
# 🔐 LOGIN PAGE
# ----------------------------
@app.route("/login")
def login():
    if is_logged_in():
        return redirect(url_for("index"))
    return render_template("login.html")

# ----------------------------
# 🔓 LOGOUT PAGE
# ----------------------------
@app.route("/logout")
def logout():
    response = make_response(redirect(url_for("login")))
    response.set_cookie("firebase_token", "", expires=0, path="/")
    return response

# ----------------------------
# 📂 MAIN PORTAL
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            uploaded_files = request.files.getlist("input_folder")
            if not uploaded_files or uploaded_files == [None]:
                return jsonify({"status": "error", "message": "No files uploaded."})

            input_folder = TEMP_EXTRACT / "current_upload"
            if input_folder.exists():
                shutil.rmtree(input_folder)
            input_folder.mkdir(parents=True, exist_ok=True)

            # Save and extract files
            for f in uploaded_files:
                filename = secure_filename(f.filename)
                saved_path = input_folder / filename
                f.save(saved_path)
                if zipfile.is_zipfile(saved_path):
                    extract_subfolder = input_folder / Path(filename).stem
                    extract_subfolder.mkdir(parents=True, exist_ok=True)
                    shutil.unpack_archive(str(saved_path), str(extract_subfolder))
                    saved_path.unlink()

            # Run your function
            if OUTPUT_FOLDER.exists():
                shutil.rmtree(OUTPUT_FOLDER)
            OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

            main_function(str(input_folder), str(OUTPUT_FOLDER))

            # Zip results
            result_zip = OUTPUT_FOLDER / "results.zip"
            if result_zip.exists():
                result_zip.unlink()
            shutil.make_archive(str(result_zip).replace(".zip", ""), "zip", OUTPUT_FOLDER)

            return jsonify({"status": "success", "download_filename": result_zip.name})

        except Exception as e:
            return jsonify({"status": "error", "message": f"Server error: {e}"})

    return render_template("index.html")

# ----------------------------
# ⬇️ DOWNLOAD ROUTE
# ----------------------------
@app.route("/download/<filename>")
def download_file(filename):
    file_path = OUTPUT_FOLDER / filename
    if not file_path.exists():
        return jsonify({"status": "error", "message": "File not found."})
    return send_file(file_path, as_attachment=True)

# ----------------------------
# 🚀 RUN APP
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))


# import os
# import shutil
# import zipfile
# from pathlib import Path
# from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session
# from werkzeug.utils import secure_filename
# from flask_cors import CORS  # ✅ CORS support

# # --- Import your analysis function ---
# from bashfile.bashfile import main_function  

# # --- Flask setup ---
# app = Flask(__name__)
# app.secret_key = os.environ.get("FLASK_SECRET_KEY", "super_secret_key_change_me")  # 🔒 Use env variable in prod

# # Enable CORS for frontend access (replace with your frontend URL in production)
# CORS(app, origins=os.environ.get("FRONTEND_URL", "*"))  

# # --- Define directories ---
# BASE_DIR = Path(__file__).resolve().parent
# UPLOAD_FOLDER = BASE_DIR / "uploads"
# OUTPUT_FOLDER = BASE_DIR / "output"
# TEMP_EXTRACT = BASE_DIR / "temp"

# for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, TEMP_EXTRACT]:
#     folder.mkdir(exist_ok=True)

# # --- Secure credentials (set via environment variables) ---
# VALID_USERNAME = os.environ.get("AKU_USER")
# VALID_PASSWORD = os.environ.get("AKU_PASS")

# if not VALID_USERNAME or not VALID_PASSWORD:
#     raise ValueError("❌ Environment variables AKU_USER and AKU_PASS must be set before running the app.")

# # ----------------------------
# # 🔐 LOGIN ROUTES
# # ----------------------------
# @app.route("/login", methods=["GET", "POST"])
# def login():
#     if request.method == "POST":
#         username = request.form.get("username")
#         password = request.form.get("password")

#         if username == VALID_USERNAME and password == VALID_PASSWORD:
#             session["user"] = username
#             return redirect(url_for("index"))
#         else:
#             return render_template("login.html", error="Invalid username or password")

#     return render_template("login.html")

# @app.route("/logout")
# def logout():
#     session.pop("user", None)
#     return redirect(url_for("login"))

# # ----------------------------
# # 📂 MAIN UPLOAD / ANALYSIS PAGE
# # ----------------------------
# @app.route("/", methods=["GET", "POST"])
# def index():
#     if "user" not in session:
#         return redirect(url_for("login"))

#     if request.method == "POST":
#         try:
#             uploaded_files = request.files.getlist("input_folder")
#             if not uploaded_files or uploaded_files == [None]:
#                 return jsonify({"status": "error", "message": "No files uploaded."})

#             # Create a unique input folder for this batch
#             input_folder = TEMP_EXTRACT / "current_upload"
#             if input_folder.exists():
#                 shutil.rmtree(input_folder)
#             input_folder.mkdir(parents=True, exist_ok=True)

#             # Save uploaded files, and extract ZIPs if needed
#             for f in uploaded_files:
#                 filename = secure_filename(f.filename)
#                 saved_path = input_folder / filename
#                 f.save(saved_path)

#                 if zipfile.is_zipfile(saved_path):
#                     extract_subfolder = input_folder / Path(filename).stem
#                     extract_subfolder.mkdir(parents=True, exist_ok=True)
#                     shutil.unpack_archive(str(saved_path), str(extract_subfolder))
#                     saved_path.unlink()  # remove ZIP after extraction

#             # Clean old outputs
#             if OUTPUT_FOLDER.exists():
#                 shutil.rmtree(OUTPUT_FOLDER)
#             OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

#             # Run analysis
#             try:
#                 main_function(str(input_folder), str(OUTPUT_FOLDER))
#             except Exception as e:
#                 return jsonify({"status": "error", "message": f"Processing error: {e}"})

#             # Create result zip
#             result_zip = OUTPUT_FOLDER / "results.zip"
#             if result_zip.exists():
#                 result_zip.unlink()
#             shutil.make_archive(str(result_zip).replace(".zip", ""), 'zip', OUTPUT_FOLDER)

#             return jsonify({
#                 "status": "success",
#                 "download_filename": result_zip.name
#             })

#         except Exception as e:
#             return jsonify({"status": "error", "message": f"Server error: {e}"})

#     return render_template("index.html")

# # ----------------------------
# # ⬇️ DOWNLOAD ROUTE
# # ----------------------------
# @app.route("/download/<filename>")
# def download_file(filename):
#     if "user" not in session:
#         return redirect(url_for("login"))

#     file_path = OUTPUT_FOLDER / filename
#     if not file_path.exists():
#         return jsonify({"status": "error", "message": "File not found."})
#     return send_file(file_path, as_attachment=True)

# # ----------------------------
# # 🚀 RUN APP
# # ----------------------------
# if __name__ == "__main__":
#     # host="0.0.0.0" makes it accessible from other devices/ngrok
#     app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))


