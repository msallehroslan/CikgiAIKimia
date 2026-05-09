import sys
sys.path.insert(0, 'api')
import os
os.environ['FIREBASE_CREDENTIALS_PATH'] = 'api/firebase_credentials.json'

try:
    from memory import init_db
    print("Firebase imported OK")
except Exception as e:
    print("Import error:", e)