import boto3
import os
import zipfile
import shutil
from urllib.parse import urlparse

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
        print("bucket_name: ", self.bucket_name)
        self.s3_key = parsed_url.path.lstrip('/')

        # Create the S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key
        )

    def fetch_and_extract(self, local_dir, folder_name):
        """
        Fetches a zip file from S3, extracts it, and stores it locally.
        
        :param local_dir: str, local directory where the extracted files will be stored
        """
        # Ensure the local directory exists
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        extraction_path = os.path.join(local_dir, folder_name)
        # Check if the directory is empty
        if os.path.exists(extraction_path):
            print(f"Data already exists in {extraction_path}, skipping download.")
            return

        # Local path to store the downloaded zip file
        local_zip_path = os.path.join(local_dir, 'data.zip')

        # Download the zip file from S3
        self.s3_client.download_file(self.bucket_name, self.s3_key, local_zip_path)

        # Extract the zip file
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(local_dir)

        # Clean up by removing the downloaded zip file
        os.remove(local_zip_path)

        # Remove the __MACOSX directory if it exists
        macosx_dir = os.path.join(local_dir, '__MACOSX')
        if os.path.exists(macosx_dir):
            shutil.rmtree(macosx_dir)

        print(f'Data fetched and extracted to {local_dir}')

