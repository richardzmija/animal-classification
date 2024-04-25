import os
import shutil
import random
import math

def split_images(source_folder, train_folder, validate_folder, test_folder):
    # Sprawdzenie i stworzenie folderów, jeśli nie istnieją
    for folder in [train_folder, validate_folder, test_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Pobranie listy podfolderów
    subfolders = [f.path for f in os.scandir(source_folder) if f.is_dir()]

    for subfolder in subfolders:
        subfolder_name = os.path.basename(subfolder)
        print("Przetwarzanie folderu:", subfolder_name)

        # Pobranie listy plików w podfolderze
        files = [f.path for f in os.scandir(subfolder) if f.is_file()]

        # Obliczenie liczby plików dla każdego zbioru danych
        num_files = len(files)
        train_size = math.ceil(0.7 * num_files)
        validate_size = math.ceil(0.2 * num_files)
        test_size = num_files - train_size - validate_size

        # Losowe przetasowanie plików
        random.shuffle(files)

        # Kopiowanie plików do odpowiednich folderów
        for i, file_path in enumerate(files):
            if i < train_size:
                destination_folder = os.path.join(train_folder, subfolder_name)
            elif i < train_size + validate_size:
                destination_folder = os.path.join(validate_folder, subfolder_name)
            else:
                destination_folder = os.path.join(test_folder, subfolder_name)

            # Stworzenie podfolderów w folderach docelowych, jeśli nie istnieją
            if not os.path.exists(destination_folder):
                os.makedirs(destination_folder)

            # Kopiowanie pliku
            shutil.copy(file_path, destination_folder)

        print("Przetwarzanie folderu", subfolder_name, "zakończone.")

if __name__ == "__main__":
    source_folder = "unprocessed"
    train_folder = "data/train"
    validate_folder = "data/validate"
    test_folder = "data/test"
    split_images(source_folder, train_folder, validate_folder, test_folder)
    print("Podział danych zakończony.")
