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