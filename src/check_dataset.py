import os

dataset_path = "dataset"

for person in os.listdir(dataset_path):

    person_path = os.path.join(dataset_path, person)

    if os.path.isdir(person_path):

        count = len([
            f for f in os.listdir(person_path)
            if f.endswith((".jpg", ".jpeg", ".png"))
        ])

        print(f"{person}: {count} images")