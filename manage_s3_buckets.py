from pathlib import Path
import json
import boto3
import shutil
import tempfile


class S3_bucket_crud:

    def __init__(
        self,
        bucket_name,
        aws_access_key_id,
        aws_secret_access_key,
        region_name,
    ):
        self.bucket_name = bucket_name

        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.region_name = region_name

        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.region_name,
        )

    # =========================================================
    # CREATE BUCKET
    # =========================================================

    def create_bucket_if_not_exists(self):

        try:
            response = self.s3_client.list_buckets()

            buckets = [
                bucket["Name"]
                for bucket in response["Buckets"]
            ]

            if self.bucket_name in buckets:
                print(
                    f"Bucket '{self.bucket_name}' already exists."
                )
                return

            self.s3_client.create_bucket(
                Bucket=self.bucket_name
            )

            print(
                f"Bucket '{self.bucket_name}' created successfully."
            )

        except Exception as e:
            print(
                f"Error creating bucket: {e}"
            )

    # =========================================================
    # UPLOAD MODEL FOLDER
    # =========================================================

    def upload_folder(self, model_dir):

        try:
            model_folder_path = Path(model_dir)

            # Use the model directory name as the S3 prefix.
            #
            # Example:
            #
            # /models/20260816_033347
            #
            # becomes:
            #
            # 20260816_033347/
            #
            s3_folder_path = model_folder_path.name

            for file_path in model_folder_path.rglob("*"):

                if not file_path.is_file():
                    continue

                relative_path = file_path.relative_to(
                    model_folder_path
                )

                s3_key = (
                    f"{s3_folder_path}/"
                    f"{relative_path}"
                )

                self.s3_client.upload_file(
                    str(file_path),
                    self.bucket_name,
                    s3_key,
                )

                print(
                    f"Uploaded: {file_path} "
                    f"to s3://{self.bucket_name}/{s3_key}"
                )

        except Exception as e:

            print(
                f"Error uploading folder: {e}"
            )

    # =========================================================
    # CREATE S3 FOLDER
    #
    # NOTE:
    # S3 does not actually require folders to be created.
    # This method is retained for compatibility, but
    # upload_folder() no longer needs to call it.
    # =========================================================

    def create_folder_in_s3(
        self,
        bucket_name,
        folder_name,
    ):

        try:

            folder_list = self.list_folders(
                bucket_name
            )

            folder_prefix = (
                f"{folder_name}/"
            )

            if folder_prefix not in folder_list:

                self.s3_client.put_object(
                    Bucket=bucket_name,
                    Key=folder_prefix,
                )

                print(
                    f"Folder '{folder_name}' created "
                    f"in bucket '{bucket_name}'."
                )

            else:

                print(
                    f"Folder '{folder_name}' already exists "
                    f"in bucket '{bucket_name}'."
                )

        except Exception as e:

            print(
                f"Error creating folder: {e}"
            )

    # =========================================================
    # LIST S3 FOLDERS
    # =========================================================

    def list_folders(
        self,
        bucket_name,
        prefix="",
    ):

        paginator = self.s3_client.get_paginator(
            "list_objects_v2"
        )

        folders = []

        for page in paginator.paginate(
            Bucket=bucket_name,
            Prefix=prefix,
            Delimiter="/",
        ):

            for common_prefix in page.get(
                "CommonPrefixes",
                []
            ):

                folders.append(
                    common_prefix["Prefix"]
                )

        return folders

    # =========================================================
    # DOWNLOAD MODEL FOLDER
    # =========================================================

    def download_folder(
        self,
        s3_folder_path,
        local_folder_path,
    ):

        temp_folder = None

        try:

            print(
                f"Downloading folder from S3: "
                f"s3://{self.bucket_name}/{s3_folder_path} "
                f"to {local_folder_path}"
            )

            local_folder_path = Path(
                local_folder_path
            )

            # -------------------------------------------------
            # Make sure S3 prefix ends with /
            # -------------------------------------------------

            if not s3_folder_path.endswith("/"):
                s3_folder_path += "/"

            # -------------------------------------------------
            # Create temporary directory
            #
            # IMPORTANT:
            # We do NOT delete current_model yet.
            # -------------------------------------------------

            temp_folder = Path(
                tempfile.mkdtemp(
                    prefix=(
                        f"{local_folder_path.name}."
                    ),
                    dir=local_folder_path.parent,
                )
            )

            print(
                f"Temporary download directory: "
                f"{temp_folder}"
            )

            paginator = (
                self.s3_client.get_paginator(
                    "list_objects_v2"
                )
            )

            downloaded_files = 0

            # -------------------------------------------------
            # Download all objects
            # -------------------------------------------------

            for page in paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=s3_folder_path,
            ):

                contents = page.get(
                    "Contents",
                    []
                )

                print(
                    f"Processing page. "
                    f"Objects found: {len(contents)}"
                )

                for obj in contents:

                    s3_key = obj["Key"]

                    # -------------------------------------------------
                    # CRITICAL FIX
                    #
                    # S3 folder markers look like:
                    #
                    # 20260816_033347/
                    #
                    # They are NOT actual files.
                    # -------------------------------------------------

                    if s3_key.endswith("/"):

                        print(
                            f"Skipping directory marker: "
                            f"{s3_key}"
                        )

                        continue

                    print(
                        f"Downloading object: "
                        f"{s3_key}"
                    )

                    # -------------------------------------------------
                    # Remove S3 folder prefix
                    #
                    # Example:
                    #
                    # 20260816_033347/config.json
                    #
                    # becomes:
                    #
                    # config.json
                    # -------------------------------------------------

                    relative_key = s3_key[
                        len(s3_folder_path):
                    ]

                    if not relative_key:

                        continue

                    relative_path = Path(
                        relative_key
                    )

                    local_file_path = (
                        temp_folder /
                        relative_path
                    )

                    # -------------------------------------------------
                    # Create parent directories
                    # -------------------------------------------------

                    local_file_path.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    # -------------------------------------------------
                    # Download file
                    # -------------------------------------------------

                    self.s3_client.download_file(
                        self.bucket_name,
                        s3_key,
                        str(local_file_path),
                    )

                    downloaded_files += 1

                    print(
                        f"Downloaded: "
                        f"s3://{self.bucket_name}/{s3_key}"
                        f" -> {local_file_path}"
                    )

            # -------------------------------------------------
            # Make sure we actually downloaded files
            # -------------------------------------------------

            if downloaded_files == 0:

                print(
                    "No model files were downloaded."
                )

                shutil.rmtree(
                    temp_folder,
                    ignore_errors=True,
                )

                temp_folder = None

                return False

            print(
                f"Download completed. "
                f"Total files downloaded: "
                f"{downloaded_files}"
            )

            # -------------------------------------------------
            # Validate config.json
            #
            # A Hugging Face model directory should contain
            # config.json.
            # -------------------------------------------------

            config_file = (
                temp_folder /
                "config.json"
            )

            if not config_file.is_file():

                print(
                    "ERROR: Downloaded model does "
                    "not contain config.json."
                )

                print(
                    f"Expected: {config_file}"
                )

                shutil.rmtree(
                    temp_folder,
                    ignore_errors=True,
                )

                temp_folder = None

                return False

            # -------------------------------------------------
            # Validate config.json contents
            # -------------------------------------------------

            try:

                with open(
                    config_file,
                    "r",
                    encoding="utf-8",
                ) as file:

                    config = json.load(file)

                if "model_type" not in config:

                    print(
                        "ERROR: config.json does not "
                        "contain 'model_type'."
                    )

                    shutil.rmtree(
                        temp_folder,
                        ignore_errors=True,
                    )

                    temp_folder = None

                    return False

                print(
                    "Model validation successful."
                )

                print(
                    f"Model type: "
                    f"{config['model_type']}"
                )

            except Exception as e:

                print(
                    f"ERROR reading config.json: {e}"
                )

                shutil.rmtree(
                    temp_folder,
                    ignore_errors=True,
                )

                temp_folder = None

                return False

            # -------------------------------------------------
            # IMPORTANT:
            #
            # Only now replace current_model.
            # -------------------------------------------------

            if local_folder_path.exists():

                print(
                    f"Removing existing model: "
                    f"{local_folder_path}"
                )

                shutil.rmtree(
                    local_folder_path
                )

            # -------------------------------------------------
            # Move completed model into final location
            # -------------------------------------------------

            temp_folder.rename(
                local_folder_path
            )

            temp_folder = None

            print(
                f"Model successfully installed at: "
                f"{local_folder_path}"
            )

            # -------------------------------------------------
            # Display downloaded files
            # -------------------------------------------------

            print(
                "Downloaded model files:"
            )

            for file_path in (
                local_folder_path.rglob("*")
            ):

                if file_path.is_file():

                    print(
                        file_path
                    )

            return True

        except Exception as e:

            print(
                f"Error downloading folder: {e}"
            )

            # -------------------------------------------------
            # Clean up failed temporary download
            # -------------------------------------------------

            if temp_folder is not None:

                shutil.rmtree(
                    temp_folder,
                    ignore_errors=True,
                )

            return False

    # =========================================================
    # READ FILE FROM S3
    # =========================================================

    def read_file_from_s3(
        self,
        filename,
    ):

        try:

            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=filename,
            )

            accuracy_json_file = json.loads(
                response["Body"]
                .read()
                .decode("utf-8")
            )

            current_model_path = (
                accuracy_json_file
                .get("current", {})
                .get("model_path")
            )

            current_accuracy = (
                accuracy_json_file
                .get("current", {})
                .get("accuracy")
            )

            if not current_model_path:

                return (
                    None,
                    None,
                    current_accuracy,
                )

            directory_name = (
                Path(
                    current_model_path
                ).name
            )

            return (
                current_model_path,
                directory_name,
                current_accuracy,
            )

        except Exception as e:

            print(
                f"Error reading file from S3: {e}"
            )

            return (
                None,
                None,
                None,
            )

    # =========================================================
    # UPLOAD SINGLE FILE
    # =========================================================

    def upload_file_to_s3(
        self,
        local_file_path,
        s3_folder_path=None,
    ):

        try:

            local_file_path = Path(
                local_file_path
            )

            if s3_folder_path:

                s3_key = (
                    f"{s3_folder_path}/"
                    f"{local_file_path.name}"
                )

            else:

                s3_key = (
                    local_file_path.name
                )

            self.s3_client.upload_file(
                str(local_file_path),
                self.bucket_name,
                s3_key,
            )

            print(
                f"Uploaded: {local_file_path} "
                f"to s3://{self.bucket_name}/{s3_key}"
            )

        except Exception as e:

            print(
                f"Error uploading file to S3: {e}"
            )