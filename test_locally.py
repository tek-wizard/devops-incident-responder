import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_flow():
    print("🔍 Step 1: Testing /tasks endpoint...")
    tasks = requests.get(f"{BASE_URL}/tasks").json()
    print(f"✅ Found {len(tasks['tasks'])} tasks.\n")

    print("🚀 Step 2: Starting 'service_restart' (Easy)...")
    # Capture the full response
    obs_res = requests.post(f"{BASE_URL}/reset?task_id=service_restart").json()
    
    # Debug: Print the keys to see what the server actually sent
    # print(f"DEBUG keys: {obs_res.keys()}") 

    # Accessing based on our SystemObservation model
    status = obs_res.get('service_status', {}).get('auth-api', 'UNKNOWN')
    error_rate = obs_res.get('metrics', {}).get('error_rate', 'MISSING')
    
    print(f"📡 Initial Status: {status}")
    print(f"📊 Initial Error Rate: {error_rate}")

    print("\n🛠️ Step 3: Sending 'restart' action...")
    action = {"command": "restart", "target": "auth-api"}
    res = requests.post(f"{BASE_URL}/step", json=action).json()
    
    # Step returns a dictionary with 'observation', 'reward', etc.
    new_obs = res["observation"]
    print(f"📡 New Status: {new_obs['service_status']['auth-api']}")
    print(f"📊 New Error Rate: {new_obs['metrics']['error_rate']}")
    print(f"📝 Latest Log: {new_obs['logs'][-1]}")

    print("\n🏆 Step 4: Checking Grader Score...")
    score_res = requests.get(f"{BASE_URL}/grader").json()
    print(f"⭐ Final Score: {score_res['score']}")

if __name__ == "__main__":
    test_flow()