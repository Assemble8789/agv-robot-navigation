import subprocess
import time
import os

def test_agv_multi():
    map_file = "maps/map_20260113_153602.json"
    if not os.path.exists(map_file):
        print(f"Error: map file {map_file} not found.")
        return

    # Start agv_world.py process
    # Use unbuffered stdout for easier reading if needed
    process = subprocess.Popen(
        ['python', 'src/agv_world.py', map_file],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    print("Started agv_world.py. Sending commands...")

    # Define 4 cars with different tasks
    commands = [
        "add car01 LM001\n",
        "add car02 LM002\n",
        "add car03 LM003\n",
        "add car04 LM006\n",
        "move car01 LM005\n",
        "move car02 LM004\n",
        "move car03 LM010\n",
        "move car04 LM011\n"
    ]

    try:
        # Give the process a moment to initialize its GUI
        time.sleep(2)

        for cmd in commands:
            print(f"Sending: {cmd.strip()}")
            process.stdin.write(cmd)
            process.stdin.flush()
            time.sleep(0.5) # Small delay between commands

        print("\nAll 4 cars have been added. They should be moving now.")
        print("You can observe the visualization. The script will wait 15 seconds then exit.")
        time.sleep(15)

        # Try to exit gracefully
        print("Exiting...")
        process.stdin.write("exit\n")
        process.stdin.flush()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        # Give it a moment to close properly
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

if __name__ == "__main__":
    test_agv_multi()
