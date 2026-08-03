from pathlib import Path
import json
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

    def upload_folder(self, model_dir):
        try:
            s3_folder_path = Path(model_dir).name
            self.create_folder_in_s3(self.bucket_name, s3_folder_path)
            model_folder_path = Path(model_dir)
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
    def download_folder(self, s3_folder_path, local_folder_path):
        try:
            from pathlib import Path
            import shutil

            print(
                f"Downloading folder from S3: "
                f"s3://{self.bucket_name}/{s3_folder_path} "
                f"to {local_folder_path}"
            )

            local_folder_path = Path(local_folder_path)

            # Remove old model files if they exist
            if local_folder_path.exists():
                print(f"Removing existing folder: {local_folder_path}")
                shutil.rmtree(local_folder_path)

            local_folder_path.mkdir(
                parents=True,
                exist_ok=True
            )

            paginator = self.s3_client.get_paginator(
                'list_objects_v2'
            )

            downloaded_files = 0

            for page in paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=s3_folder_path
            ):

                print(
                    f"Processing page. "
                    f"Objects found: {page.get('KeyCount', 0)}"
                )

                contents = page.get('Contents', [])

                if not contents:
                    print(
                        f"No objects found for prefix: {s3_folder_path}"
                    )
                    continue

                for obj in contents:

                    s3_key = obj["Key"]

                    print(
                        f"Downloading object: {s3_key}"
                    )

                    # Remove the S3 folder prefix
                    relative_path = Path(s3_key).relative_to(
                        s3_folder_path
                    )

                    local_file_path = (
                        local_folder_path / relative_path
                    )

                    # Create parent directories
                    local_file_path.parent.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    self.s3_client.download_file(
                        self.bucket_name,
                        s3_key,
                        str(local_file_path)
                    )

                    downloaded_files += 1

                    print(
                        f"Downloaded: "
                        f"s3://{self.bucket_name}/{s3_key}"
                        f" -> {local_file_path}"
                    )

            if downloaded_files == 0:
                print(
                    "No files were downloaded."
                )
                return False

            print(
                f"Download completed. "
                f"Total files downloaded: {downloaded_files}"
            )

            print("Downloaded model files:")

            for file in local_folder_path.rglob("*"):
                if file.is_file():
                    print(file)

            return True

        except Exception as e:
            print(
                f"Error downloading folder: {e}"
            )
            return False
    def read_file_from_s3(self, filename):
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=filename)
            accuracy_json_file = json.loads(response['Body'].read().decode('utf-8'))
            current_model_path = accuracy_json_file.get('current', {}).get('model_path')
            current_accuracy = accuracy_json_file.get('current', {}).get('accuracy')
            directory_name = Path(current_model_path).name
            return current_model_path, directory_name, current_accuracy
        except Exception as e:
            print(f"Error reading file from S3: {e}")
            return None
    def upload_file_to_s3(self, local_file_path, s3_folder_path=None):
        try:
            local_file_path = Path(local_file_path)
            if s3_folder_path:
                s3_key = f"{s3_folder_path}/{local_file_path.name}"
            else:
                s3_key = local_file_path.name
            self.s3_client.upload_file(str(local_file_path), self.bucket_name, s3_key)
            print(f"Uploaded: {local_file_path} to s3://{self.bucket_name}/{s3_key}")
        except Exception as e:
            print(f"Error uploading file to S3: {e}")