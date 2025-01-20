def create_output_directories(base_path):
    """
    Creates necessary output directories for data img and notebooks.
    """
    folders_to_create = ['data', 'img', 'notebooks']
    list_of_folders = []
    for main_folder in folders_to_create:
        folder_path = base_path / main_folder
        folder_path.mkdir(parents=True, exist_ok=True)
        list_of_folders.append(folder_path)
    return list_of_folders

def create_data_directories(base_path):
    """
    Creates necessary output directories for data raw and processed.
    """
    folders_to_create = ['raw', 'processed']
    list_of_folders = []
    for main_folder in folders_to_create:
        folder_path = base_path / main_folder
        folder_path.mkdir(parents=True, exist_ok=True)
        list_of_folders.append(folder_path)
    return list_of_folders

def create_img_directories(base_path):
        """
        Crea los directorios necesarios para img.
        """
        folder_path = base_path / 'img'
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path