import subprocess
import sys

def main():
    print("Starting custom engine...")
    subprocess.run([sys.executable, "-m", "server.app"] + sys.argv[1:])

if __name__ == "__main__":
    main()
