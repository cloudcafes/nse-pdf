import os
import sqlite3
import json
import time
import requests
from google import genai 

# ==============================
# API KEYS & CONFIGURATION
# ==============================
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
DB_NAME            = "nse_pipeline.db"

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# ==========================================
# TELEGRAM ALERTS
# ==========================================

def send_telegram_message(content):
    if not content or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    print("    [TELEGRAM] Dispatching alert...")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    chunks = [content[i:i+4000] for i in range(0, len(content), 4000)]
    for chunk in chunks:
        try:
            requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk})
        except Exception as e:
            print(f"    [X] Telegram request error: {e}")
        time.sleep(1)

# ==========================================
# BATCH PROCESSING
# ==========================================

def process_batch_results(job_info, conn):
    cursor = conn.cursor()
    
    try:
        output_file_name = job_info.dest.file_name
        print(f"[POLL] Downloading results from internal File API: {output_file_name}...")
        
        response_bytes = client.files.download(file=output_file_name)
        results_text = response_bytes.decode("utf-8")
            
        for line in results_text.splitlines():
            if not line.strip(): continue
            data = json.loads(line)
            
            filepath = data.get("key")
            response_payload = data.get("response", {})
            
            if "candidates" in response_payload and response_payload["candidates"]:
                try:
                    result_text = response_payload["candidates"][0]["content"]["parts"][0]["text"].strip()
                    cursor.execute("UPDATE report_pipeline SET llm_status = 'SUCCESS', llm_summary = ? WHERE filepath = ?", (result_text, filepath))
                    
                    # Clean up markdown formatting (e.g., **Potential:** High) to ensure strict matching
                    result_upper = result_text.upper().replace("*", "").replace(" ", "")
                    
                    # Check for High/Very High
                    if "POTENTIAL:HIGH" in result_upper or "POTENTIAL:VERYHIGH" in result_upper:
                        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
                            print(f"    [!] ALERT BLOCKED for {filepath}: Telegram Token or Chat ID is missing in environment.")
                        else:
                            print(f"    [+] HIGH POTENTIAL DETECTED for {filepath}!")
                            send_telegram_message(result_text)
                    else:
                        print(f"    [-] Skipped alert for {filepath}: Potential is LOW/IGNORE/NA.")
                        
                except (KeyError, IndexError):
                    print(f"    [X] Parsing Error for {filepath}.")
                    cursor.execute("UPDATE report_pipeline SET llm_status = 'FAILED', llm_summary = 'Parsing Error' WHERE filepath = ?", (filepath,))
            else:
                print(f"    [X] Blocked/Empty Response for {filepath}.")
                cursor.execute("UPDATE report_pipeline SET llm_status = 'FAILED', llm_summary = 'Blocked/Empty Response' WHERE filepath = ?", (filepath,))
                
        cursor.execute("UPDATE active_batches SET status = 'COMPLETED' WHERE job_id = ?", (job_info.name,))
        conn.commit()
        print("[SUCCESS] Database updated and batch loop closed.")
        
    except Exception as e:
        print(f"[X] Result processing failed: {e}")

def check_active_batches():
    if not client or not os.path.exists(DB_NAME): return

    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        
        # Verify migration actually occurred
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='active_batches'")
        if not cursor.fetchone(): return

        cursor.execute("SELECT job_id FROM active_batches WHERE status = 'POLLING'")
        active_jobs = cursor.fetchall()
        
        if not active_jobs:
            print("[SYSTEM] No active batch jobs to poll.")
            return
            
        for (job_id,) in active_jobs:
            print(f"\n[POLL] Checking status for job: {job_id}")
            try:
                job_info = client.batches.get(name=job_id)
                
                # Safe string comparison for the JobState Enum
                current_state_name = job_info.state.name if hasattr(job_info.state, 'name') else str(job_info.state)
                print(f"  -> API State: {current_state_name}")
                
                if current_state_name == "JOB_STATE_SUCCEEDED":
                    process_batch_results(job_info, conn)
                elif current_state_name in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"):
                    print(f"  -> [!] Job {job_id} terminated unproductively.")
                    cursor.execute("UPDATE active_batches SET status = ? WHERE job_id = ?", (current_state_name, job_id))
                    cursor.execute("UPDATE report_pipeline SET llm_status = ? WHERE batch_job_id = ?", (current_state_name, job_id))
                    conn.commit()
                    
            except Exception as e:
                print(f"[X] API Error polling {job_id}: {e}")

if __name__ == "__main__":
    check_active_batches()
