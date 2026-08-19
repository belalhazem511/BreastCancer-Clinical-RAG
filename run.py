import os
import sys
import time
import socket
import argparse
import webbrowser
import threading
import uvicorn
from clean_port import kill_process_on_port

def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex((host, port)) == 0

def find_available_port(start_port: int = 8000, host: str = "127.0.0.1", max_attempts: int = 20) -> int:
    for port in range(start_port, start_port + max_attempts):
        if not is_port_in_use(port, host):
            return port
    return start_port

def open_browser(port: int):
    time.sleep(1.2)
    url = f"http://127.0.0.1:{port}"
    print(f"\n[+] Opening BreastCancer.ai in your browser: {url}\n")
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"Could not open browser automatically: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BreastCancer.ai Launcher")
    parser.add_argument("--port", type=int, default=None, help="Port to run server on (default: 8000 or next free port)")
    parser.add_argument("--clean", action="store_true", help="Force terminate any process occupying the target port before starting")
    args = parser.parse_args()

    # Determine desired port
    env_port = os.getenv("PORT")
    desired_port = args.port or (int(env_port) if env_port and env_port.isdigit() else 8000)

    # If --clean was requested or port is in use and clean is enabled
    if args.clean:
        kill_process_on_port(desired_port)

    # Check if port is in use; if so, find next free port
    if is_port_in_use(desired_port):
        print(f"[!] Port {desired_port} is currently in use.")
        free_port = find_available_port(desired_port + 1)
        print(f"[*] Automatically switching to available port: {free_port}")
        active_port = free_port
    else:
        active_port = desired_port

    print("=" * 60)
    print("   BreastCancer.ai - Clinical RAG Assistant Full Cycle   ")
    print("=" * 60)
    print(f"[*] Server Port:  {active_port}")
    print(f"[*] Access URL:   http://127.0.0.1:{active_port}")
    print(f"[*] Guidelines:   NICE NG101, NICE CG81, NICE CG164")
    print(f"[*] Status:       Ready")
    print("[*] Press Ctrl+C to stop the server.")
    print("=" * 60)

    # Open browser in a background thread
    threading.Thread(target=open_browser, args=(active_port,), daemon=True).start()

    # Run FastAPI app
    uvicorn.run("server:app", host="127.0.0.1", port=active_port, reload=False, log_level="info")
