import sqlite3

def setup_db():
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    # Create the single table
    c.execute('''
        CREATE TABLE IF NOT EXISTS board_captures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            image_path TEXT NOT NULL,
            raw_text TEXT,
            confidence_score REAL
        )
    ''')
    conn.commit()
    conn.close()
    print("Database app.db and table board_captures created safely.")

if __name__ == "__main__":
    setup_db()
