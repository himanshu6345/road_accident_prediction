import sqlite3
import json
import os
import hashlib
import secrets
import mysql.connector

DB_PATH = "app_data.db"
OLD_USERS_DB = "users.json"

DB_TYPE = os.getenv("DB_TYPE", "sqlite").lower()
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "road_accident_db")

class DBConnection:
    def __init__(self):
        self.conn = None
        self.cursor = None
        
        global DB_TYPE
        if DB_TYPE == "mysql":
            try:
                # Attempt to connect to the server without DB first to create it if needed
                temp_conn = mysql.connector.connect(
                    host=DB_HOST,
                    user=DB_USER,
                    password=DB_PASSWORD
                )
                temp_c = temp_conn.cursor()
                temp_c.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
                temp_conn.close()
                
                self.conn = mysql.connector.connect(
                    host=DB_HOST,
                    user=DB_USER,
                    password=DB_PASSWORD,
                    database=DB_NAME
                )
                self.cursor = self.conn.cursor(dictionary=True)
            except Exception as e:
                print(f"MySQL Connection Error: {e}. Falling back to SQLite.")
                self.setup_sqlite()
        else:
            self.setup_sqlite()

    def setup_sqlite(self):
        global DB_TYPE
        DB_TYPE = "sqlite"
        self.conn = sqlite3.connect(DB_PATH)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()

    def execute(self, query, params=()):
        # Handle placeholder differences
        if DB_TYPE == "mysql":
            query = query.replace('?', '%s')
        self.cursor.execute(query, params)

    def fetchone(self):
        row = self.cursor.fetchone()
        if row is None: return None
        return dict(row)

    def fetchall(self):
        rows = self.cursor.fetchall()
        return [dict(row) for row in rows]

    def commit(self):
        self.conn.commit()

    def close(self):
        self.cursor.close()
        self.conn.close()

def get_db_connection():
    return DBConnection()

def init_db():
    db = get_db_connection()
    
    # Create users table
    auto_increment = "AUTO_INCREMENT" if DB_TYPE == "mysql" else "AUTOINCREMENT"
    
    db.execute(f'''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY {auto_increment},
            username VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            salt VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    try:
        db.execute('ALTER TABLE users ADD COLUMN email VARCHAR(255)')
        db.commit()
    except Exception:
        pass # Column already exists
        
    try:
        db.execute('ALTER TABLE users ADD COLUMN full_name VARCHAR(255)')
        db.commit()
    except Exception:
        pass
        
    try:
        db.execute('ALTER TABLE users ADD COLUMN contact_number VARCHAR(50)')
        db.commit()
    except Exception:
        pass
    
    # Create predictions table
    db.execute(f'''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY {auto_increment},
            username VARCHAR(255) NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            input_features TEXT NOT NULL,
            rf_prediction VARCHAR(255),
            svm_prediction VARCHAR(255),
            FOREIGN KEY(username) REFERENCES users(username)
        )
    ''')
    
    db.commit()
    db.close()
    
    # Migrate old users if exists
    migrate_users()

def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)
    
    # create a hash
    key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
    return key.hex(), salt

def add_user(username, password, email=None, full_name=None, contact_number=None):
    db = get_db_connection()
    
    try:
        # Check if user exists
        db.execute('SELECT username FROM users WHERE username = ?', (username,))
        if db.fetchone() is not None:
            import random
            suggestions = [f"{username}_{random.randint(10,999)}", f"{username}Pro", f"{username}{random.randint(1,99)}"]
            return False, f"Username '{username}' already exists! Try: {', '.join(suggestions)}"
            
        hashed_pw, salt = hash_password(password)
        db.execute('INSERT INTO users (username, password_hash, salt, email, full_name, contact_number) VALUES (?, ?, ?, ?, ?, ?)',
                  (username, hashed_pw, salt, email, full_name, contact_number))
        db.commit()
        return True, "Account created successfully!"
    except Exception as e:
        return False, str(e)
    finally:
        db.close()

def verify_user(username, password):
    db = get_db_connection()
    
    db.execute('SELECT password_hash, salt FROM users WHERE username = ?', (username,))
    user = db.fetchone()
    db.close()
    
    if user is None:
        return False
        
    stored_hash = user['password_hash']
    salt = user['salt']
    
    attempt_hash, _ = hash_password(password, salt)
    
    return secrets.compare_digest(stored_hash, attempt_hash)

def reset_password(username, email, new_password):
    db = get_db_connection()
    
    try:
        db.execute('SELECT email FROM users WHERE username = ?', (username,))
        user = db.fetchone()
        
        if user is None:
            return False, "Username not found!"
            
        stored_email = user.get('email')
        if not stored_email or stored_email.strip().lower() != email.strip().lower():
            return False, "Verification failed: Email does not match our records."
            
        hashed_pw, salt = hash_password(new_password)
        db.execute('UPDATE users SET password_hash = ?, salt = ? WHERE username = ?',
                  (hashed_pw, salt, username))
        db.commit()
        return True, "Password reset successfully! You can now sign in."
    except Exception as e:
        return False, str(e)
    finally:
        db.close()

def migrate_users():
    if os.path.exists(OLD_USERS_DB):
        try:
            with open(OLD_USERS_DB, "r") as f:
                users = json.load(f)
            
            # Iterate and add each user if they don't already exist
            for username, password in users.items():
                add_user(username, password)
                
            # Rename the old DB so we don't migrate again
            os.rename(OLD_USERS_DB, OLD_USERS_DB + ".migrated")
        except Exception as e:
            print(f"Error migrating users: {e}")

def log_prediction(username, input_features, rf_pred, svm_pred):
    db = get_db_connection()
    try:
        db.execute('''
            INSERT INTO predictions (username, input_features, rf_prediction, svm_prediction)
            VALUES (?, ?, ?, ?)
        ''', (username, json.dumps(input_features), rf_pred, svm_pred))
        db.commit()
    except Exception as e:
        print(f"Error logging prediction: {e}")
    finally:
        db.close()

def get_predictions(username):
    db = get_db_connection()
    
    db.execute('''
        SELECT timestamp, input_features, rf_prediction, svm_prediction 
        FROM predictions 
        WHERE username = ? 
        ORDER BY timestamp DESC
    ''', (username,))
    
    records = db.fetchall()
    db.close()
    
    return records

def get_all_users():
    db = get_db_connection()
    
    db.execute('''
        SELECT id, username, created_at
        FROM users
        ORDER BY created_at DESC
    ''')
    
    records = db.fetchall()
    db.close()
    
    return records

def get_all_predictions():
    db = get_db_connection()
    
    db.execute('''
        SELECT username, timestamp, input_features, rf_prediction, svm_prediction
        FROM predictions
        ORDER BY timestamp DESC
    ''')
    
    records = db.fetchall()
    db.close()
    
    return records
