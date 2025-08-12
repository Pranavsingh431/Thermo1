#!/usr/bin/env python3
"""
Quick script to create test users without starting the server
"""

import sys
import os
from pathlib import Path
import sqlite3
from passlib.context import CryptContext

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def create_users():
    """Create test users directly in SQLite database"""
    print("🔧 Creating test users...")

    # Connect to SQLite database
    db_path = "thermal_inspection.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Create admin user
        admin_password = hash_password("admin123")
        cursor.execute("""
            INSERT OR REPLACE INTO users
            (username, email, full_name, employee_id, department, designation, role, hashed_password, is_active, is_verified)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "admin",
            "admin@tatapower.com",
            "System Administrator",
            "TP_ADMIN",
            "IT",
            "System Admin",
            "admin",
            admin_password,
            True,
            True
        ))

        # Create engineer user
        engineer_password = hash_password("engineer123")
        cursor.execute("""
            INSERT OR REPLACE INTO users
            (username, email, full_name, employee_id, department, designation, role, hashed_password, is_active, is_verified)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "engineer",
            "engineer@tatapower.com",
            "Senior Electrical Engineer",
            "TP_ENG001",
            "Electrical",
            "Senior Engineer",
            "engineer",
            engineer_password,
            True,
            True
        ))

        conn.commit()
        print("✅ Users created successfully!")
        print("📝 Login credentials:")
        print("   Admin: username=admin, password=admin123")
        print("   Engineer: username=engineer, password=engineer123")

    except Exception as e:
        print(f"❌ Error creating users: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    create_users()
