import boto3
import os
import zipfile
import shutil
from urllib.parse import urlparse
import tempfile

class S3DataFetcher:
    def __init__(self, aws_access_key_id, aws_secret_access_key, s3_uri):
        """
        Initialize the S3DataFetcher class.
        
        :param aws_access_key_id: str, AWS access key ID
        :param aws_secret_access_key: str, AWS secret access key
        :param s3_uri: str, S3 URI (e.g., s3://bucket-name/path/to/data.zip)
        """
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.s3_uri = s3_uri

        # Parse the S3 URI to get the bucket name and key
        parsed_url = urlparse(s3_uri)
        self.bucket_name = parsed_url.netloc
        self.s3_key = parsed_url.path.lstrip('/')

        # Create the S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key
        )

    def fetch_and_extract(self, local_dir, folder_name):
        """
        Fetches a zip file from S3, extracts it to a tmpfs directory (/dev/shm), 
        and moves the cleaned-up data to the target directory.
        
        :param local_dir: str, local directory where the extracted files will be stored
        :param folder_name: str, name of the folder to store extracted data
        """
        # Ensure the local directory exists
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        extraction_path = os.path.join(local_dir, folder_name)

        # Check if the directory is empty
        if os.path.exists(extraction_path):
            print(f"Data already exists in {extraction_path}, skipping download.")
            return

        # Use /dev/shm (tmpfs) for temporary storage to avoid using root filesystem
        tmpfs_dir = '/dev/shm'
        temp_dir = tempfile.mkdtemp(dir=tmpfs_dir)
        
        try:
            local_zip_path = os.path.join(temp_dir, 'data.zip')

            # Download the zip file from S3 to the tmpfs directory
            self.s3_client.download_file(self.bucket_name, self.s3_key, local_zip_path)

            # Extract the zip file to the tmpfs directory
            with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            # Remove the __MACOSX directory if it exists in the tmpfs directory
            macosx_dir = os.path.join(temp_dir, '__MACOSX')
            if os.path.exists(macosx_dir):
                shutil.rmtree(macosx_dir)

            # Move the cleaned extracted data from tmpfs to the final destination
            shutil.move(temp_dir, extraction_path)

        finally:
            # Clean up the temporary directory if it still exists
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

        print(f'Data fetched and extracted to {extraction_path}')
