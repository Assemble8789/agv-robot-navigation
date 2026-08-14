import requests
import time

BASE_URL = "http://localhost:8001"

def test_api():
    print("Testing AGV World REST API...")
    
    # 1. Add 4 cars
    add_cars = [
        {"carID": "car01", "landmark": "LM001"},
        {"carID": "car02", "landmark": "LM002"},
        {"carID": "car03", "landmark": "LM003"},
        {"carID": "car04", "landmark": "LM006"}
    ]
    
    for car in add_cars:
        print(f"Adding {car['carID']}...")
        try:
            r = requests.post(f"{BASE_URL}/add", json=car)
            print(f"Response: {r.json()}")
        except Exception as e:
            print(f"Failed to add car: {e}")
        time.sleep(0.5)

    # 2. Start moving them
    move_cars = [
        {"carID": "car01", "goalLM": "LM005"},
        {"carID": "car02", "goalLM": "LM004"},
        {"carID": "car03", "goalLM": "LM010"},
        {"carID": "car04", "goalLM": "LM011"}
    ]
    for move in move_cars:
        print(f"Moving {move['carID']} to {move['goalLM']}...")
        try:
            r = requests.post(f"{BASE_URL}/move", json=move)
            print(f"Response: {r.json()}")
        except Exception as e:
            print(f"Failed to move car: {e}")
        time.sleep(0.5)

    # 2. Polling status
    for _ in range(5):
        time.sleep(3)
        try:
            r = requests.get(f"{BASE_URL}/status")
            status = r.json()
            print(f"Time: {status['world_time']} | Cars: {len(status['cars'])}")
        except Exception as e:
            print(f"Failed to get status: {e}")

    # 3. Move car01 after it stops
    # Note: This might fail if car01 is still moving, we could check status first
    print("Requesting car01 to move to LM009...")
    move_data = {"carID": "car01", "goalLM": "LM009"}
    try:
        r = requests.post(f"{BASE_URL}/move", json=move_data)
        print(f"Move response: {r.json()}")
    except Exception as e:
        print(f"Move error: {e}")

if __name__ == "__main__":
    test_api()
