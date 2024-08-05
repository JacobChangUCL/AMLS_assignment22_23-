
import subprocess
import time


def run_python_files_sequentially(files):
    for file in files:
        time.sleep(5)
        try:
            process = subprocess.Popen(['python', file], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            # 实时输出 stdout
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(f"{file} output: {output.strip()}")
            # 捕获并输出 stderr
            stderr = process.stderr.read()
            if stderr:
                print(f"Error in {file}:\n{stderr.strip()}")
            process.wait()
        except Exception as e:
            print(f"Error running {file}: {e}")

files_to_run = ["A1/gender_detection.py","A2/emotion_detection.py","B1/face_shape_recognition.py","B2/eye_color_recognition.py"]

run_python_files_sequentially(files_to_run)