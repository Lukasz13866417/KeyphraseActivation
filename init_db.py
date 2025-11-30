import db.db_api as db_api
import sys
try:
    db_api.init_db()
except Exception as e:
    print(f"Error initializing database: {e}")
    sys.exit(1)