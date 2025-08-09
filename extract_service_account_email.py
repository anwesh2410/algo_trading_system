#!/usr/bin/env python3
"""
Extract service account email from Google credentials JSON file
"""
import json
import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from config.config_loader import Config

def main():
    """Extract and display service account email"""
    # Get credentials file path
    creds_file = Config.GOOGLE_SHEETS_CREDENTIALS_FILE
    
    if not creds_file or not os.path.exists(creds_file):
        print("Enter path to Google credentials file:")
        creds_file = input("> ").strip()
        
        if not os.path.exists(creds_file):
            print(f"❌ ERROR: File not found: {creds_file}")
            return
    
    try:
        # Read JSON file
        with open(creds_file, 'r') as f:
            creds_data = json.load(f)
        
        # Extract email
        if 'client_email' in creds_data:
            email = creds_data['client_email']
            print("\n==================================================")
            print(f"Service Account Email: {email}")
            print("==================================================")
            print("\nIMPORTANT: You must share your Google Sheet with this email address!")
            print("\nFollow these steps:")
            print("1. Open your Google Sheet at this URL:")
            print(f"   https://docs.google.com/spreadsheets/d/{Config.GOOGLE_SHEETS_SPREADSHEET_ID}")
            print("2. Click the 'Share' button in the top-right corner")
            print("3. Paste the service account email in the 'Add people' field")
            print("4. Set permission to 'Editor'")
            print("5. UNCHECK 'Notify people' option")
            print("6. Click 'Share'")
            print("7. Run your test script again")
        else:
            print("❌ ERROR: 'client_email' not found in credentials file")
    
    except Exception as e:
        print(f"❌ ERROR: Failed to parse credentials file: {str(e)}")

if __name__ == "__main__":
    main()