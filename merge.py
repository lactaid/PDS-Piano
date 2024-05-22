import os
import shutil

def combine_folders(data_folder, old_data_folder, combined_folder):
    # Crear la carpeta combinada si no existe
    if not os.path.exists(combined_folder):
        os.makedirs(combined_folder)

    # Iterar sobre las carpetas numeradas del 1 al 12
    for folder_num in range(1, 13):
        data_path = os.path.join(data_folder, str(folder_num))
        old_data_path = os.path.join(old_data_folder, str(folder_num))

        combined_path = os.path.join(combined_folder, str(folder_num))
        # Crear la carpeta combinada para cada número si no existe
        if not os.path.exists(combined_path):
            os.makedirs(combined_path)

        # Copiar imágenes de data a la carpeta combinada
        for filename in os.listdir(data_path):
            if filename.endswith('.jpg'):
                shutil.copy(os.path.join(data_path, filename), combined_path)

        # Copiar imágenes de old_data a la carpeta combinada, renombrándolas
        for filename in os.listdir(old_data_path):
            if filename.endswith('.jpg'):
                old_image_path = os.path.join(old_data_path, filename)
                new_image_name = str(int(filename[:-4]) + 1500) + '.jpg'  # Renombrar con nuevo número
                new_image_path = os.path.join(combined_path, new_image_name)
                shutil.copy(old_image_path, new_image_path)

if __name__ == "__main__":
    base_folder = "model"
    data_folder = os.path.join(base_folder, "data")
    old_data_folder = os.path.join(base_folder, "old_data")
    combined_folder = os.path.join(base_folder, "hand_data")

    combine_folders(data_folder, old_data_folder, combined_folder)
    print("Proceso completado. Las carpetas han sido combinadas en 'model/hand_data'.")
