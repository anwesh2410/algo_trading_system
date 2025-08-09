import requests
import sys

def test_telegram_bot(bot_token, chat_id):
    """Test Telegram bot connectivity"""
    print(f"Testing Telegram bot with token: {bot_token[:5]}...{bot_token[-5:]}")
    print(f"Testing with chat ID: {chat_id}")
    
    # First, get bot information to verify token
    try:
        response = requests.get(f"https://api.telegram.org/bot{bot_token}/getMe")
        if response.status_code == 200:
            bot_data = response.json()
            if bot_data["ok"]:
                bot_name = bot_data["result"]["username"]
                print(f"✅ Bot token is valid. Connected to @{bot_name}")
            else:
                print(f"❌ Error: {bot_data['description']}")
                return False
        else:
            print(f"❌ Error: HTTP status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {str(e)}")
        return False
        
    # Now test sending a message to verify chat ID
    try:
        message = "This is a test message from Algo Trading System."
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'Markdown'
        }
        
        response = requests.post(f"https://api.telegram.org/bot{bot_token}/sendMessage", data=payload)
        
        if response.status_code == 200:
            data = response.json()
            if data["ok"]:
                print("✅ Test message sent successfully!")
                return True
            else:
                print(f"❌ Error: {data['description']}")
                return False
        else:
            print(f"❌ Error: HTTP status code {response.status_code}")
            print(f"Response: {response.text}")
            
            if "chat not found" in response.text.lower():
                print("\nTROUBLESHOOTING TIPS FOR CHAT ID:")
                print("1. Make sure you've started a conversation with your bot")
                print("2. To get your chat ID:")
                print("   a. Send a message to your bot")
                print("   b. Visit https://api.telegram.org/bot{YOUR_BOT_TOKEN}/getUpdates")
                print("   c. Look for 'chat':{'id': YOUR_CHAT_ID}")
            
            return False
    except Exception as e:
        print(f"❌ Message sending error: {str(e)}")
        return False

if __name__ == "__main__":
    print("====== TELEGRAM BOT TEST ======")
    bot_token = input("Enter your Telegram bot token: ")
    chat_id = input("Enter your chat ID: ")
    
    if test_telegram_bot(bot_token, chat_id):
        print("\n✅ All tests passed! Your Telegram bot is working correctly.")
        print(f"Add these to your .env file:")
        print(f"TELEGRAM_BOT_TOKEN={bot_token}")
        print(f"TELEGRAM_CHAT_ID={chat_id}")
    else:
        print("\n❌ Tests failed. Please check the errors above.")