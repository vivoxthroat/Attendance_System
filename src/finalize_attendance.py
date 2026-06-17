import pandas as pd

ALL_STUDENTS = [
    "Vivek",
    "Anushka"
]

attendance_file = "attendance/attendance.csv"

df = pd.read_csv(attendance_file)

present_students = set(df["Name"])

for student in ALL_STUDENTS:

    if student not in present_students:

        df.loc[len(df)] = [
            student,
            "Absent",
            "-",
            "Neutral"
        ]

df.to_csv(
    attendance_file,
    index=False
)

print("Attendance finalized.")