import os
import glob
import io
import urllib.request
import sqlite3
from datetime import datetime
from flask import Flask, render_template, jsonify, send_from_directory, send_file, request
from fpdf import FPDF
import requests

try:
    from rag_engine import RAGEngine
    rag = RAGEngine()
except Exception as e:
    rag = None
    print(f"[RAG WARNING] Failed to boot RAGEngine, Chat functions disabled locally. Error: {e}")

app = Flask(__name__)
CAPTURE_DIR = "captures"
FONT_PATH = "DejaVuSans.ttf"

def ensure_font_loaded():
    if not os.path.exists(FONT_PATH):
        print("[System] Downloading DejaVuSans.ttf for PDF Unicode support...")
        try:
            url = "https://raw.githubusercontent.com/prawnpdf/prawn/master/data/fonts/DejaVuSans.ttf"
            urllib.request.urlretrieve(url, FONT_PATH)
        except Exception as e:
            print(f"[Warning] Failed to download Unicode font: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/pages')
def get_pages():
    conn = sqlite3.connect('app.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    # Simple Pagination using SQL
    limit = request.args.get('limit', type=int)
    offset = request.args.get('offset', default=0, type=int)
    
    if limit is not None:
        c.execute("SELECT * FROM board_captures ORDER BY id ASC LIMIT ? OFFSET ?", (limit, offset))
    else:
        c.execute("SELECT * FROM board_captures ORDER BY id ASC OFFSET ?", (offset,))
        
    rows = c.fetchall()
    conn.close()
    
    pages = []
    for row in rows:
        # Standardize formatting to provide robust REST mapping
        base_name = os.path.splitext(os.path.basename(row['image_path']))[0]
        text_content = row['raw_text'] if row['raw_text'] else ""
        
        pages.append({
            'id': base_name,
            'image_url': f'/captures/{base_name}.jpg',
            'text': text_content,
            'confidence': row['confidence_score'],
            'processing': False
        })
        
    return jsonify(pages)

@app.route('/api/search')
def api_search():
    query = request.args.get('q', type=str)
    if not query:
        return jsonify([])
        
    conn = sqlite3.connect('app.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM board_captures WHERE raw_text LIKE ?", ('%' + query + '%',))
    rows = c.fetchall()
    conn.close()
    
    results = [dict(row) for row in rows]
    return jsonify(results)

@app.route('/api/chat', methods=['POST'])
def api_chat():
    if not rag:
        return jsonify({"answer": "Intelligence API is disconnected offline.", "sources": []})

    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing query payload"}), 400
        
    query = data['query']
    
    # 1. Pre-sync specific vector state locally
    conn = sqlite3.connect('app.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM board_captures")
    rows = c.fetchall()
    conn.close()
    
    rag.sync_db_to_vector_store([dict(r) for r in rows])
    
    # 2. Extract context dimensions
    matches = rag.retrieve_context(query, top_k=3)
    if not matches:
        return jsonify({"answer": "I don't have any contextual insights on that topic.", "sources": []})
        
    # Format structural LLM Prompt exactly handling fuzzy OCR formatting
    context_text = "\n---\n".join([f"Board Scan #{i+1}: {m['text']}" for i, m in enumerate(matches)])
    sources = [m['metadata'] for m in matches]
    
    system_prompt = f"""You are an advanced Expert Architecture Teaching Assistant interpreting extracted OCR.
The human text represents chaotic whiteboards, and may contain fragments, typographical errors or mathematical inaccuracies (e.g. "t(x) - x2" instead of "f(x) = x^2").
CRITICAL RULE: Reconstruct logical flows based ONLY on logical interpretations of the text. Do NOT hallucinate mathematical theorems or concepts NOT strictly found in the provided OCR blocks.
Answer honestly relying strictly on the following board snapshots:

<CONTEXT>
{context_text}
</CONTEXT>"""

    payload = {
        "model": "llama3.1:8b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }
    
    # 3. Call local lightweight JSON post processing
    try:
        response = requests.post('http://localhost:11434/api/chat', json=payload, timeout=25)
        response.raise_for_status()
        answer = response.json()['message']['content']
    except Exception as e:
        answer = f"[LLM REST ERROR] Ensure Ollama is running structurally on port 11434! {str(e)}"
        
    return jsonify({
        "answer": answer,
        "sources": sources
    })

@app.route('/captures/<filename>')
def serve_capture(filename):
    return send_from_directory(CAPTURE_DIR, filename)

@app.route('/export')
def export_pdf():
    conn = sqlite3.connect('app.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM board_captures ORDER BY id ASC")
    rows = c.fetchall()
    conn.close()
    
    if not rows:
        return "No captures found currently generated by the OCR engine.", 404

    ensure_font_loaded()

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    if os.path.exists(FONT_PATH):
        pdf.add_font("DejaVu", "", FONT_PATH)
        base_font = "DejaVu"
    else:
        base_font = "helvetica"
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    for row in rows:
        pdf.add_page()
        # Header
        try:
             pdf.set_font(base_font, "", 16)
        except Exception:
             pdf.set_font("helvetica", "B", 16)
        pdf.cell(0, 10, "Smart Board Capture", align="C", new_x="LMARGIN", new_y="NEXT")
        
        try:
             pdf.set_font(base_font, "", 10)
        except Exception:
             pdf.set_font("helvetica", "I", 10)
        pdf.cell(0, 10, f"Recorded on: {timestamp}", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        
        # Max width 180mm to fit standard A4 margin
        pdf.image(row['image_path'], w=180)
        pdf.ln(10)
        
        try:
             pdf.set_font(base_font, "", 12)
        except Exception:
             pdf.set_font("helvetica", "", 12)
             
        content = row['raw_text'] if row['raw_text'] else "[OCR Read Incomplete]"
            
        if base_font == "helvetica":
            content = content.encode('latin-1', 'replace').decode('latin-1') 
            
        pdf.multi_cell(0, 8, content)
            
    # Volatile RAM export
    pdf_bytes = pdf.output()
    return send_file(io.BytesIO(pdf_bytes), as_attachment=True, download_name="Lecture_Notes.pdf", mimetype='application/pdf')

def run_flask():
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

if __name__ == '__main__':
    run_flask()
