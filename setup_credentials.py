# Update the setup_google_sheets function
def setup_google_sheets():
    """Setup Google Sheets credentials"""
    print("\n===== Google Sheets Setup =====")
    print("To use Google Sheets integration, you need to:")
    print("1. Create a Google Cloud project")
    print("2. Enable Google Sheets API and Drive API")
    print("3. Create a service account and download credentials")
    
    # Check if credentials directory exists
    creds_dir = Path(__file__).parent / "credentials"
    creds_dir.mkdir(exist_ok=True)
    
    # Ask for credentials file
    print("\nDo you have a Google API credentials JSON file? (y/n)")
    has_creds = input("> ").strip().lower() == "y"
    
    if has_creds:
        print("\nPlease enter the path to your credentials JSON file:")
        creds_path = input("> ").strip()
        
        if os.path.exists(creds_path):
            # Copy to credentials directory
            target_path = creds_dir / "google_sheets_credentials.json"
            shutil.copy(creds_path, target_path)
            
            # Update .env file
            update_env_file("GOOGLE_SHEETS_CREDENTIALS_FILE", str(target_path))
            
            print(f"\n✅ Credentials copied to {target_path}")
            print("✅ .env file updated with credentials path")
            
            # Ask for spreadsheet ID or name
            print("\nDo you want to use an existing Google Sheet? (y/n)")
            use_existing = input("> ").strip().lower() == "y"
            
            if use_existing:
                print("\nEnter the Google Sheet ID (found in the sheet URL):")
                sheet_id = input("> ").strip()
                if sheet_id:
                    update_env_file("GOOGLE_SHEETS_SPREADSHEET_ID", sheet_id)
                    print(f"✅ Spreadsheet ID set to: {sheet_id}")
            else:
                print("\nEnter the name for your Google Sheet (default: AlgoTradingSystem):")
                sheet_name = input("> ").strip()
                if not sheet_name:
                    sheet_name = "AlgoTradingSystem"
                    
                update_env_file("GOOGLE_SHEETS_SPREADSHEET_NAME", sheet_name)
                print(f"✅ Spreadsheet name set to: {sheet_name}")
            
        else:
            print(f"\n❌ File not found: {creds_path}")
            print("Please check the file path and try again.")
    else:
        print("\nPlease follow these steps to get your credentials:")
        print("1. Go to https://console.cloud.google.com/")
        print("2. Create a new project")
        print("3. Enable Google Sheets API and Google Drive API")
        print("4. Create a service account")
        print("5. Create and download a JSON key")
        print("6. Run this setup script again")