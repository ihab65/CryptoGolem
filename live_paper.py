import os
import time
import requests
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Binance Testnet API URL
TESTNET_URL = "https://testnet.binance.vision"

# Load API keys (Make sure these are testnet keys)
API_KEY = os.getenv("PAPER_TRADING")
API_SECRET = os.getenv("PAPER_TRADING_SECRET")

def create_binance_client(max_retries=5, initial_delay=3):
    """
    Creates and returns a Binance client with retry logic.
    Handles connection issues, API failures, and implements exponential backoff.
    """
    retry_delay = initial_delay

    for attempt in range(1, max_retries + 1):
        try:
            client = Client(API_KEY, API_SECRET, testnet=True)
            client.API_URL = TESTNET_URL  # Ensure testnet endpoint
            client.ping()  # Check if the connection is live
            print("✅ Connected to Binance Testnet successfully.")
            return client  # Return the working client
        
        except requests.ConnectionError as e:
            print(f"🌐 Connection failed (Attempt {attempt}/{max_retries}): {e}")
        except requests.Timeout as e:
            print(f"⏳ Timeout error (Attempt {attempt}/{max_retries}): {e}")
        except BinanceAPIException as e:
            print(f"🚨 Binance API error (Attempt {attempt}/{max_retries}): {e}")
        except requests.RequestException as e:
            print(f"⚠️ General request error (Attempt {attempt}/{max_retries}): {e}")

        # If max attempts reached, exit
        if attempt == max_retries:
            print("❌ Failed to establish connection after multiple attempts.")
            return None

        # Exponential backoff
        print(f"🔄 Retrying in {retry_delay} seconds...")
        time.sleep(retry_delay)
        retry_delay *= 2

# Initialize the client
client = create_binance_client()

# Check if connection succeeded before proceeding
if client:
    # Fetch and print USDT balance
    usdt_balance = client.get_asset_balance(asset="USDT")
    print(f"\n🔹 USDT Balance: {usdt_balance['free']}")
    client.get_trade_fee()
else:
    print("⚠️ Could not connect to Binance. Check your internet and API keys.")

