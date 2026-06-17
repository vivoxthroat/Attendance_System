import pandas as pd
from datetime import datetime
import os

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_FILE = os.path.join(BASE_DIR, "attendance", "attendance.csv")

def mark_attendance(name, emotion):

    os.makedirs("attendance", exist_ok=True)

    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
    else:
        df = pd.DataFrame(
            columns=["Name", "Status", "Time", "Emotion"]
        )

    if name in df["Name"].values:
        return

    new_row = {
        "Name": name,
        "Status": "Present",
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Emotion": emotion
    }

    df = pd.concat(
        [df, pd.DataFrame([new_row])],
        ignore_index=True
    )

    df.to_csv(CSV_FILE, index=False)

    print(f"{name} marked present")