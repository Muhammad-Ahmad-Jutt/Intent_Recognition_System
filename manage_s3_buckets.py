from pathlib import Path

import boto3

class S3_bucket_crud:
    def __init__(
        self,
        bucket_name,
        aws_access_key_id,
        aws_secret_access_key,
        region_name,
        endpoint_url=None
    ):
        self.bucket_name = bucket_name

        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.region_name = region_name
        self.endpoint_url = endpoint_url

        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.region_name,
            endpoint_url=self.endpoint_url
        )

    def create_bucket_if_not_exists(self):
        try:
            response = self.s3_client.list_buckets()

            buckets = [
                bucket['Name']
                for bucket in response['Buckets']
            ]

            if self.bucket_name in buckets:
                print(f"Bucket '{self.bucket_name}' already exists.")
                return

            self.s3_client.create_bucket(
                Bucket=self.bucket_name
            )

            print(f"Bucket '{self.bucket_name}' created successfully.")

        except Exception as e:
            print(f"Error creating bucket: {e}")

    def upload_folder(self, model_out_directory, s3_folder_path):
        try:
            model_folder_path = Path(model_out_directory)
            for files in model_folder_path.rglob('*'):
                if files.is_file():
                    relative_path = files.relative_to(model_folder_path)
                    s3_key = f"{s3_folder_path}/{relative_path}"
                    self.s3_client.upload_file(str(files), self.bucket_name, s3_key)
                    print(f"Uploaded: {files} to s3://{self.bucket_name}/{s3_key}")

        except Exception as e:
            print(f"Error uploading folder: {e}")
    def create_folder_in_s3(self, bucket_name, folder_name):
        try:
            folder_list = self.list_folders(bucket_name)
            if folder_name not in folder_list:
                self.s3_client.put_object(Bucket=bucket_name, Key=f"{folder_name}/")
                print(f"Folder '{folder_name}' created in bucket '{bucket_name}'.")
            else:
                print(f"Folder '{folder_name}' already exists in bucket '{bucket_name}'.")
        except Exception as e:
            print(f"Error creating folder: {e}")
    def list_folders(self, bucket_name, prefix=''):
        s3_client = boto3.client('s3')
        paginator = s3_client.get_paginator('list_objects_v2')
        folders = []
        
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter='/'):
            for cp in page.get('CommonPrefixes', []):
                folders.append(cp['Prefix'])
                
        return folders
    def download_folder(self, bucket_name, s3_folder_path, local_folder_path):
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder_path):
                for obj in page.get('Contents', []):
                    s3_key = obj['Key']
                    relative_path = Path(s3_key).relative_to(s3_folder_path)
                    local_file_path = Path(local_folder_path) / relative_path
                    local_file_path.parent.mkdir(parents=True, exist_ok=True)
                    self.s3_client.download_file(bucket_name, s3_key, str(local_file_path))
                    print(f"Downloaded: s3://{bucket_name}/{s3_key} to {local_file_path}")
        except Exception as e:
            print(f"Error downloading folder: {e}")