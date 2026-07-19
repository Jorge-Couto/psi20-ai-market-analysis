import subprocess
import sys

notebooks = [
    "Data_Downloads.ipynb",
    "Time_Series_Forecasting_Final.ipynb",
    "AI_Agents.ipynb"
]

for nb in notebooks:
    print(f"\n==== Running {nb} ====\n")

    try:
        subprocess.run([
            sys.executable,
            "-m",
            "papermill",
            nb,
            nb,  # overwrite same notebook (no extra files)
            "--log-output"  # THIS is the key line
        ], check=True)

        print(f"\n✅ {nb} executed successfully!\n")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error executing {nb}: {e}")
        break