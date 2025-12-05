import os
from azure.storage.filedatalake import DataLakeServiceClient
from azure.core.exceptions import ResourceExistsError

STORAGE_ACCOUNT_NAME = "lab560103194"
STORAGE_ACCOUNT_KEY = "acesss-key"
CONTAINER_NAME = "lakehouse"

LOCAL_DATA_DIR = "../data/brain_tumor_dataset"

REMOTE_BASE_DIR = "raw/tumor_images"


def get_service_client():
    """Initialize and return DataLakeServiceClient"""
    account_url = f"https://{STORAGE_ACCOUNT_NAME}.dfs.core.windows.net"
    service_client = DataLakeServiceClient(
        account_url=account_url,
        credential=STORAGE_ACCOUNT_KEY
    )
    return service_client


def create_directory_structure(file_system_client):
    """Create the directory structure in ADLS Gen2"""
    directories = [
        f"{REMOTE_BASE_DIR}/yes",
        f"{REMOTE_BASE_DIR}/no"
    ]
    
    for directory in directories:
        try:
            directory_client = file_system_client.get_directory_client(directory)
            directory_client.create_directory()
            print(f"Created directory: {directory}")
        except ResourceExistsError:
            print(f"Directory already exists: {directory}")


def upload_images(file_system_client, local_category_dir, remote_category_dir):
    """Upload images from local directory to ADLS Gen2"""
    if not os.path.exists(local_category_dir):
        print(f"Local directory not found: {local_category_dir}")
        return
    
    files = os.listdir(local_category_dir)
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg'))]
    
    print(f"\nUploading {len(image_files)} images from {local_category_dir} to {remote_category_dir}...")
    
    for idx, filename in enumerate(image_files, 1):
        local_file_path = os.path.join(local_category_dir, filename)
        remote_file_path = f"{remote_category_dir}/{filename}"
        
        try:
            file_client = file_system_client.get_file_client(remote_file_path)
            
            with open(local_file_path, 'rb') as data:
                file_client.upload_data(data, overwrite=True)
            
            print(f"  [{idx}/{len(image_files)}] Uploaded: {filename}")
        except Exception as e:
            print(f"  Error uploading {filename}: {str(e)}")


def main():
    """Main function to ingest images into ADLS Gen2"""
    print("Starting image ingestion to Azure ADLS Gen2...")

    service_client = get_service_client()
    file_system_client = service_client.get_file_system_client(CONTAINER_NAME)

    print("\nCreating directory structure...")
    create_directory_structure(file_system_client)
  
    categories = ['yes', 'no']
    for category in categories:
        local_dir = os.path.join(LOCAL_DATA_DIR, category)
        remote_dir = f"{REMOTE_BASE_DIR}/{category}"
        upload_images(file_system_client, local_dir, remote_dir)
    
    print("\n✓ Image ingestion completed successfully!")


if __name__ == "__main__":
    main()
