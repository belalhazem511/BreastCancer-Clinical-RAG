import os
import sys
import subprocess
import socket

def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0

def kill_process_on_port(port: int) -> bool:
    print(f"[*] Checking for processes listening on port {port}...")
    try:
        # Use netstat to find PID
        output = subprocess.check_output(f"netstat -ano | findstr :{port}", shell=True, text=True)
        pids = set()
        for line in output.strip().split("\n"):
            parts = line.split()
            if len(parts) >= 5 and "LISTENING" in parts:
                pid = parts[-1]
                if pid.isdigit() and int(pid) > 0:
                    pids.add(int(pid))
        
        if not pids:
            print(f"[!] No active listening PID found on port {port}.")
            return False

        for pid in pids:
            print(f"[*] Terminating process PID {pid} on port {port}...")
            subprocess.run(f"taskkill /F /PID {pid}", shell=True, capture_output=True)
        
        print(f"[OK] Port {port} has been freed successfully.")
        return True
    except subprocess.CalledProcessError:
        print(f"[*] Port {port} is already free.")
        return True
    except Exception as e:
        print(f"[!] Error while freeing port {port}: {e}")
        return False

if __name__ == "__main__":
    target_port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    kill_process_on_port(target_port)
