#!/usr/bin/env python3
"""
Detailed diagnostic tool for Google Sheets connection issues
"""
import os
import sys
import json
import traceback
from pathlib import Path
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import time

def print_section(title):
    """Print a section header"""
    print("\n" + "="*60)
    print(title)
    print("="*60)

def load_credentials_file(creds_path):
    """Load and validate credentials file"""
    print_section("CHECKING CREDENTIALS FILE")
    
    # Verify file exists
    if not os.path.exists(creds_path):
        print(f"❌ ERROR: Credentials file not found at {creds_path}")
        return None
    
    print(f"✓ Found credentials file at {creds_path}")
    
    # Verify file is valid JSON
    try:
        with open(creds_path, 'r') as f:
            creds_data = json.load(f)
            print("✓ Credentials file is valid JSON")
            
            # Print key fields
            print("\nCredentials info:")
            print(f"  Project ID: {creds_data.get('project_id', 'Not found')}")
            print(f"  Client Email: {creds_data.get('client_email', 'Not found')}")
            
            # Check for required fields
            required_fields = ['type', 'project_id', 'private_key_id', 'private_key', 
                              'client_email', 'client_id', 'auth_uri', 'token_uri']
            missing_fields = [field for field in required_fields if field not in creds_data]
            
            if missing_fields:
                print(f"❌ WARNING: Missing required fields: {', '.join(missing_fields)}")
            else:
                print("✓ All required credential fields present")
                
            return creds_data
    except json.JSONDecodeError:
        print(f"❌ ERROR: Credentials file is not valid JSON")
        return None
    except Exception as e:
        print(f"❌ ERROR: Failed to read credentials file: {str(e)}")
        return None

def test_auth(creds_data, creds_path):
    """Test authentication with both oauth2client and google.oauth2"""
    print_section("TESTING AUTHENTICATION")
    
    # Test with oauth2client (older library)
    print("\nTesting with oauth2client:")
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
        print("✓ Successfully created credentials object")
        
        client = gspread.authorize(creds)
        print("✓ Successfully authorized with gspread")
        return client
    except Exception as e:
        print(f"❌ Authentication failed: {str(e)}")
        print(traceback.format_exc())
    
    # If the first method fails, try the newer google.oauth2
    print("\nTrying alternative authentication method:")
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = Credentials.from_service_account_info(creds_data, scopes=scope)
        client = gspread.authorize(creds)
        print("✓ Successfully authorized with alternative method")
        return client
    except Exception as e:
        print(f"❌ Alternative authentication failed: {str(e)}")
        print(traceback.format_exc())
        
    return None

def check_apis(creds_path):
    """Check if required APIs are enabled"""
    print_section("CHECKING API ENABLEMENT")
    
    try:
        # Use the Discovery API to check
        credentials = ServiceAccountCredentials.from_json_keyfile_name(
            creds_path, 
            ['https://www.googleapis.com/auth/cloud-platform']
        )
        
        service = build('serviceusage', 'v1', credentials=credentials)
        
        # Get project ID from credentials
        with open(creds_path, 'r') as f:
            creds_data = json.load(f)
            project_id = creds_data.get('project_id', '')
        
        if not project_id:
            print("❌ ERROR: Could not find project_id in credentials")
            return
            
        project_name = f"projects/{project_id}"
        
        # Check Google Sheets API
        try:
            request = service.services().get(name=f"{project_name}/services/sheets.googleapis.com")
            response = request.execute()
            if response.get('state') == 'ENABLED':
                print("✓ Google Sheets API is enabled")
            else:
                print("❌ Google Sheets API is NOT enabled")
        except Exception as e:
            print(f"⚠️ Could not check Sheets API status: {str(e)}")
        
        # Check Google Drive API
        try:
            request = service.services().get(name=f"{project_name}/services/drive.googleapis.com")
            response = request.execute()
            if response.get('state') == 'ENABLED':
                print("✓ Google Drive API is enabled")
            else:
                print("❌ Google Drive API is NOT enabled")
        except Exception as e:
            print(f"⚠️ Could not check Drive API status: {str(e)}")
            
    except Exception as e:
        print(f"⚠️ Could not check API enablement: {str(e)}")
        print("You should manually verify these APIs are enabled in your Google Cloud Console:")
        print("1. Google Sheets API")
        print("2. Google Drive API")

def test_spreadsheet_access(client, spreadsheet_id):
    """Test access to the specific spreadsheet"""
    print_section("TESTING SPREADSHEET ACCESS")
    
    if not client:
        print("❌ Cannot test spreadsheet access: Authentication failed")
        return
        
    print(f"Attempting to access spreadsheet ID: {spreadsheet_id}")
    
    try:
        # First list all accessible spreadsheets
        print("\nListing all accessible spreadsheets:")
        try:
            all_sheets = client.list_spreadsheet_files()
            if not all_sheets:
                print("⚠️ No spreadsheets accessible to this service account")
            else:
                print(f"✓ Found {len(all_sheets)} accessible spreadsheets:")
                for sheet in all_sheets:
                    print(f"  - {sheet['name']} (ID: {sheet['id']})")
                    if sheet['id'] == spreadsheet_id:
                        print(f"    ✓ Target spreadsheet found in accessible list!")
        except Exception as e:
            print(f"⚠️ Could not list spreadsheets: {str(e)}")
        
        # Try to open the specific spreadsheet
        print(f"\nAttempting to open spreadsheet with ID {spreadsheet_id}:")
        spreadsheet = client.open_by_key(spreadsheet_id)
        print(f"✓ Successfully opened spreadsheet: {spreadsheet.title}")
        
        # List worksheets
        print("\nListing worksheets:")
        worksheets = spreadsheet.worksheets()
        for ws in worksheets:
            print(f"  - {ws.title} (ID: {ws.id})")
        
        # Try a simple operation
        print("\nAttempting to read from first worksheet:")
        first_ws = worksheets[0]
        cell_value = first_ws.acell('A1').value
        print(f"✓ Successfully read cell A1: '{cell_value}'")
        
        # Try writing to the spreadsheet
        print("\nAttempting to write to first worksheet:")
        test_value = f"Test: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        first_ws.update_cell(1, 2, test_value)
        print(f"✓ Successfully wrote '{test_value}' to cell B1")
        
        return True
    except gspread.exceptions.APIError as e:
        print(f"❌ API Error: {str(e)}")
        if "The caller does not have permission" in str(e):
            print("\n⚠️ PERMISSION ERROR:")
            print("1. Make sure you shared the spreadsheet with EXACTLY this email:")
            with open(creds_path, 'r') as f:
                client_email = json.load(f).get('client_email', 'Unknown')
            print(f"   {client_email}")
            print("2. Make sure you gave 'Editor' permission")
            print("3. Google permissions can take a few minutes to propagate")
            print("4. Make sure the spreadsheet is not restricted to your organization")
        return False
    except Exception as e:
        print(f"❌ Error accessing spreadsheet: {str(e)}")
        print(traceback.format_exc())
        return False

def provide_recommendations():
    """Provide recommendations based on common issues"""
    print_section("RECOMMENDATIONS")
    
    print("Based on common issues, here are steps to try:")
    print("1. Double-check that you shared with the EXACT service account email")
    print("2. Wait 5 minutes for Google permissions to fully propagate")
    print("3. Verify the APIs are enabled in Google Cloud Console:")
    print("   - Google Sheets API")
    print("   - Google Drive API")
    print("4. Try creating a new spreadsheet and sharing it with the service account")
    print("5. Make sure your JSON credentials file is complete and valid")
    print("6. If using a Google Workspace account, check if external sharing is restricted")
    
    print("\nTo create a new test spreadsheet:")
    print("1. Go to https://sheets.google.com/")
    print("2. Create a new blank spreadsheet")
    print("3. Share it with your service account email")
    print("4. Copy the ID from the URL")
    print("5. Update your .env file with the new ID")

if __name__ == "__main__":
    print("Google Sheets Connection Diagnostic Tool")
    print("---------------------------------------")
    
    # Get credentials path
    creds_path = input("Enter path to your Google credentials JSON file: ")
    creds_path = creds_path.strip()
    
    # Get spreadsheet ID
    spreadsheet_id = input("Enter your Google Sheet ID (from the URL): ")
    spreadsheet_id = spreadsheet_id.strip()
    
    # Load and check credentials
    creds_data = load_credentials_file(creds_path)
    if not creds_data:
        print("\n❌ Cannot proceed: Invalid credentials file")
        sys.exit(1)
    
    # Test authentication
    client = test_auth(creds_data, creds_path)
    if not client:
        print("\n❌ Cannot proceed: Authentication failed")
        provide_recommendations()
        sys.exit(1)
    
    # Check APIs
    check_apis(creds_path)
    
    # Test spreadsheet access
    success = test_spreadsheet_access(client, spreadsheet_id)
    
    if success:
        print_section("SUCCESS")
        print("✓ All tests passed! Your Google Sheets integration is working correctly.")
        print("\nAdd these lines to your .env file:")
        print(f"GOOGLE_SHEETS_CREDENTIALS_FILE={creds_path}")
        print(f"GOOGLE_SHEETS_SPREADSHEET_ID={spreadsheet_id}")
    else:
        print_section("FAILED")
        print("❌ Tests failed. Please check the error messages above.")
        provide_recommendations()