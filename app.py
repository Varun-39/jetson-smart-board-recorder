import os
import glob
from datetime import datetime
from flask import Flask, render_template, jsonify, send_from_directory, send_file
from fpdf import FPDF

app = Flask(__name__)
CAPTURE_DIR = "captures"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/pages')
def get_pages():
    if not os.path.exists(CAPTURE_DIR):
        return jsonify([])
    
    # Find all .jpg files
    images = glob.glob(os.path.join(CAPTURE_DIR, "*.jpg"))
    # Sort them sequentially based on the page number
    images.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    pages = []
    for img_path in images:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(CAPTURE_DIR, f"{base_name}.txt")
        
        text_content = ""
        processing = False
        
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
        else:
            # If the image exists but txt doesn't, OCR thread is still working
            text_content = "Processing..."
            processing = True
            
        pages.append({
            'id': base_name,
            'image_url': f'/captures/{base_name}.jpg',
            'text': text_content,
            'processing': processing
        })
        
    return jsonify(pages)

@app.route('/captures/<filename>')
def serve_capture(filename):
    return send_from_directory(CAPTURE_DIR, filename)

@app.route('/export')
def export_pdf():
    if not os.path.exists(CAPTURE_DIR):
        return "No captures found", 404
        
    images = glob.glob(os.path.join(CAPTURE_DIR, "*.jpg"))
    images.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    if not images:
        return "No captures found", 404

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    for img_path in images:
        pdf.add_page()
        # Header
        pdf.set_font("helvetica", "B", 16)
        pdf.cell(0, 10, "Smart Board Capture", align="C", new_x="LMARGIN", new_y="NEXT") # Updated to fpdf2 syntax
        pdf.set_font("helvetica", "I", 10)
        pdf.cell(0, 10, f"Recorded on: {timestamp}", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        
        # Max width 180mm to fit standard A4 margin
        pdf.image(img_path, w=180)
        pdf.ln(10)
        
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(CAPTURE_DIR, f"{base_name}.txt")
        
        pdf.set_font("helvetica", "", 12)
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Encode text replacing unprintable chars since fpdf struggles with unicode out of the box
                content = content.encode('latin-1', 'replace').decode('latin-1') 
                pdf.multi_cell(0, 8, content)
        else:
            pdf.multi_cell(0, 8, "[OCR Processing Incomplete]")
            
    pdf_path = "Lecture_Notes.pdf"
    pdf.output(pdf_path)
    
    return send_file(pdf_path, as_attachment=True)

def run_flask():
    # Keep console silent using werkzeug logger
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

if __name__ == '__main__':
    run_flask()
