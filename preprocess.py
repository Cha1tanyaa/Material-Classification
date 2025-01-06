from datasets import load_dataset, DatasetDict

# Lade den Datensatz und wähle den 'train'-Split aus
ds = load_dataset("garythung/trashnet")
ds.save_to_disk("/home/lkracht/Dokumente/Vorlesungen/5_Semester/Computer_Vision_for_Explainable_AI/Projekt/Computer-Vision")
# Teile den Datensatz in Training und Test (z. B. 80% Training, 20% Test)
'''train_test_split = ds.train_test_split(test_size=0.2, seed=42)

# Teile das Trainings-Set weiter in Training und Validierung (z. B. 75% Training, 25% Validierung)
train_val_split = train_test_split["train"].train_test_split(test_size=0.25, seed=42)

# Ergebnis zusammenfügen
split_ds = DatasetDict({
    "train": train_val_split["train"],
    "validation": train_val_split["test"],
    "test": train_test_split["test"]
})

# Preprocessing-Funktion für Bilder und Labels
def preprocess_data(data):
    images, labels = [], []
    for i, image in enumerate(data["image"]):
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
            tensor_image = tf.convert_to_tensor(tf.keras.preprocessing.image.img_to_array(image))
            resized_image = tf.image.resize(tensor_image, (150, 150)) / 255.0
            images.append(resized_image)
        except Exception as e:
            print(f"Skipping image {i} due to error: {e}")
            continue
    labels = tf.convert_to_tensor(data["label"][:len(images)], dtype=tf.int32)  # Sync labels with valid images
    return {"image": tf.stack(images), "label": labels}

# Wende Preprocessing auf alle Splits an
split_ds = {key: ds.map(preprocess_data, batched=True, batch_size=10) for key, ds in split_ds.items()}

# Speichere den Datensatz auf die Festplatte
split_ds.save_to_disk("/home/lkracht/Dokumente/Vorlesungen/5_Semester/Computer_Vision_for_Explainable_AI/Projekt/Computer-Vision")
'''
