import os
import time
import sqlite3
import fitz  # PyMuPDF
from datetime import datetime
from curl_cffi import requests as cffi_requests
import requests  # Standard requests for Telegram API
import google.generativeai as genai

# ==============================
# API KEYS & CONFIGURATION
# ==============================
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_MODEL       = "gemini-3.1-flash-lite-preview"

DB_NAME = "nse_pipeline.db"
BASE_URL = "https://www.nseindia.com"
ARCHIVE_URL = "https://nsearchives.nseindia.com"
ANALYSIS_FILE = "latest-analysis.txt"

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL)

# ==========================================
# 1. DATABASE & EXTRACTION COMPONENT
# ==========================================

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS report_pipeline (
            filepath TEXT PRIMARY KEY,
            symbol TEXT,
            download_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            pdf_text TEXT, 
            llm_status TEXT DEFAULT 'PENDING',
            llm_summary TEXT
        )
    ''')
    conn.commit()
    conn.close()

def extract_pdf_text(filepath):
    print(f"    [PROCESS] Extracting text from {os.path.basename(filepath)}...")
    text_content = ""
    try:
        with fitz.open(filepath) as doc:
            for page in doc:
                text_content += page.get_text()
        return text_content
    except Exception as e:
        print(f"    [X] PDF Extraction Failed: {e}")
        return None

def register_download_and_text(filepath, symbol, pdf_text):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR IGNORE INTO report_pipeline (filepath, symbol, pdf_text, llm_status)
        VALUES (?, ?, ?, 'PENDING')
    ''', (filepath, symbol, pdf_text))
    conn.commit()
    conn.close()

def get_pending_text_for_llm():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT filepath, symbol, pdf_text FROM report_pipeline 
        WHERE llm_status IN ('PENDING', 'FAILED') AND pdf_text IS NOT NULL
    ''')
    results = cursor.fetchall()
    conn.close()
    return results

def update_llm_status(filepath, status, summary=None):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    if summary:
        cursor.execute('UPDATE report_pipeline SET llm_status = ?, llm_summary = ? WHERE filepath = ?', (status, summary, filepath))
    else:
        cursor.execute('UPDATE report_pipeline SET llm_status = ? WHERE filepath = ?', (status, filepath))
    conn.commit()
    conn.close()

def print_db_summary():
    if not os.path.exists(DB_NAME): return
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    print("\n" + "="*50 + "\n📊 DATABASE PIPELINE SUMMARY\n" + "="*50)
    try:
        cursor.execute("SELECT COUNT(*) FROM report_pipeline")
        print(f"Total Documents Tracked: {cursor.fetchone()[0]}")
        cursor.execute("SELECT COUNT(*) FROM report_pipeline WHERE pdf_text IS NOT NULL")
        print(f"Documents with Extracted Text: {cursor.fetchone()[0]}")
        
        print("\n--- LLM Processing Status ---")
        cursor.execute("SELECT llm_status, COUNT(*) FROM report_pipeline GROUP BY llm_status")
        status_dict = {status: count for status, count in cursor.fetchall()}
        print(f"  ✅ SUCCESS (Analyzed): {status_dict.get('SUCCESS', 0)}")
        print(f"  ⏳ PENDING (Waiting):  {status_dict.get('PENDING', 0)}")
        print(f"  ❌ FAILED  (Errors):   {status_dict.get('FAILED', 0)}")
    except Exception as e:
        pass
    print("="*50 + "\n")
    conn.close()

# ==========================================
# 2. AI & TELEGRAM INTEGRATION
# ==========================================

def analyze_with_gemini(symbol, pdf_text):
    """Sends the extracted text to Gemini using the strict Prompt."""
    prompt = f"""You are a professional equity research analyst specializing in event-driven trading strategies.
Your task is to analyze a corporate announcement and determine whether it creates a short-term trading opportunity.
You MUST extract and infer the following fields:
1. Company Name
2. Reason to Trade:
   - Explain in 1–2 concise lines WHY this announcement may impact stock price
   - If no actionable insight → return "NA"
3. Date:
   - Extract from the announcement or metadata
   - Format: DD-MM-YYYY
4. Potential:
   Classify into ONE of the following:
   - VERYHIGH → Major price-moving event (earnings surprise, M&A, large order win, regulatory approval, stake sale, etc.)
   - HIGH → Strong positive/negative signal but not transformational
   - LOW → Minor impact, informational
   - IGNORE → Routine filings (compliance, disclosures, minor updates)
   - NA → If unclear
---
## Analysis Guidelines (IMPORTANT)
- Focus ONLY on **market-moving signals**
- Ignore boilerplate language
- Detect:
  - Earnings results (beat/miss)
  - Large contracts/orders
  - M&A / acquisitions
  - Promoter activity
  - Fundraising / dilution
  - Regulatory actions
  - Credit rating changes
  - Guidance / outlook changes
- DO NOT hallucinate
- If information is insufficient → output NA fields
---
## Output Format (STRICT — no extra text)
Return ONLY in this format:
Company Name: <value>
Reason to Trade: <value or NA>
Date: <DD-MM-YYYY or NA>
Potential: <VERYHIGH | HIGH | LOW | IGNORE | NA>
---
## Input Announcement:
<<<
{pdf_text}
>>>"""
    try:
        response = model.generate_content(prompt)
        return True, response.text.strip()
    except Exception as e:
        print(f"    [X] Gemini API Error: {e}")
        return False, str(e)

def send_telegram_message(filepath):
    """Reads the analysis file and sends it to Telegram, chunking if necessary."""
    if not os.path.exists(filepath):
        return
        
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()
        
    if not content:
        return
        
    print("[SYSTEM] Sending analysis to Telegram...")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    # Telegram max message length is 4096 chars. Safely chunking at 4000.
    chunks = [content[i:i+4000] for i in range(0, len(content), 4000)]
    
    for chunk in chunks:
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": chunk}
        try:
            response = requests.post(url, json=payload)
            if response.status_code != 200:
                print(f"    [X] Telegram failed: {response.text}")
        except Exception as e:
            print(f"    [X] Telegram request error: {e}")
        time.sleep(1) # Prevent Telegram API rate limits

# ==========================================
# 3. CORE PIPELINE WORKFLOW
# ==========================================

def run_pipeline():
    init_db()
    today_str = datetime.now().strftime("%d-%m-%Y")
    download_dir = f"NSE_Reports_{today_str}"
    os.makedirs(download_dir, exist_ok=True)
    
    # Secure session initialization
    session = cffi_requests.Session(impersonate="chrome120")
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Referer": f"{BASE_URL}/companies-listing/corporate-filings-announcements"
    })
    
    print("[SYSTEM] Fetching session cookies & API Data...")
    try:
        session.get(BASE_URL, timeout=15)
        time.sleep(2)
        api_url = f"{BASE_URL}/api/corporate-announcements?index=equities&from_date={today_str}&to_date={today_str}"
        response = session.get(api_url, timeout=15)
        data = response.json()
    except Exception as e:
        print(f"[ERROR] API or Fetch failed: {e}")
        return

    # --- PHASE 1: DOWNLOAD & EXTRACT ---
    for item in data:
        att_path = item.get("attchmntFile") or item.get("attchmntText") or item.get("att")
        symbol = item.get("symbol", "UNKNOWN")
        
        if att_path:
            att_path_str = str(att_path).strip()
            if '.pdf' in att_path_str.lower() or '/corporate/' in att_path_str.lower():
                pdf_url = att_path_str if att_path_str.startswith("http") else f"{ARCHIVE_URL}{att_path_str}"
                safe_filename = att_path_str.split('/')[-1].split('?')[0] 
                if not safe_filename.lower().endswith('.pdf'): safe_filename += ".pdf"
                filepath = os.path.join(download_dir, f"{symbol}_{safe_filename}")
                
                if not os.path.exists(filepath):
                    print(f"\n  [DOWNLOAD] Fetching {symbol} -> {safe_filename}...")
                    try:
                        pdf_response = session.get(pdf_url, timeout=20)
                        if pdf_response.status_code == 200:
                            with open(filepath, 'wb') as f:
                                f.write(pdf_response.content)
                            time.sleep(2) 
                    except Exception as e:
                        print(f"    [X] Request Error: {e}")
                        continue
                
                conn = sqlite3.connect(DB_NAME)
                cursor = conn.cursor()
                cursor.execute('SELECT 1 FROM report_pipeline WHERE filepath = ?', (filepath,))
                already_in_db = cursor.fetchone()
                conn.close()

                if not already_in_db:
                    extracted_text = extract_pdf_text(filepath)
                    if extracted_text:
                        register_download_and_text(filepath, symbol, extracted_text)

    # --- PHASE 2: LLM ANALYSIS & TELEGRAM ---
    unprocessed = get_pending_text_for_llm()
    
    if unprocessed:
        print(f"\n[SYSTEM] Found {len(unprocessed)} files ready for AI analysis...")
        
        # 1. Clear out the old analysis file for this run
        if os.path.exists(ANALYSIS_FILE):
            os.remove(ANALYSIS_FILE)
            
        # 2. Run the AI Loop
        success_count = 0
        for filepath, symbol, pdf_text in unprocessed:
            print(f"  [AI] Querying Gemini for {symbol}...")
            
            success, result = analyze_with_gemini(symbol, pdf_text)
            
            if success:
                # Append the strict output to our text file
                with open(ANALYSIS_FILE, "a", encoding="utf-8") as f:
                    f.write(f"{result}\n\n{'='*40}\n\n")
                    
                update_llm_status(filepath, 'SUCCESS', result)
                success_count += 1
            else:
                update_llm_status(filepath, 'FAILED')
                
            # Rate limit protection for Gemini API
            time.sleep(3) 

        # 3. Fire to Telegram
        if success_count > 0:
            send_telegram_message(ANALYSIS_FILE)
            
    else:
        print("\n[SYSTEM] No new documents pending AI analysis.")

    print_db_summary()

if __name__ == "__main__":
    run_pipeline()