import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv


load_dotenv(Path(__file__).parent.parent / ".env")

S3_ENDPOINT = "https://storage.yandexcloud.net"
S3_BUCKET   = os.environ["S3_BUCKET"]
S3_PREFIX   = "second/nlp/weights/"
WEIGHTS_DIR = Path(__file__).parent.parent / "weights"

WEIGHTS_DIR.mkdir(exist_ok=True)

env = os.environ.copy()
env["AWS_ACCESS_KEY_ID"]     = os.environ["YC_ACCESS_KEY_ID"]
env["AWS_SECRET_ACCESS_KEY"] = os.environ["YC_SECRET_ACCESS_KEY"]

subprocess.run(
    f'aws s3 sync s3://{S3_BUCKET}/{S3_PREFIX} "{WEIGHTS_DIR}" --endpoint-url {S3_ENDPOINT}',
    env=env,
    shell=True,
)

subprocess.run(
    f'aws s3 rm s3://{S3_BUCKET}/{S3_PREFIX} --recursive --exclude "*" --include "*" --endpoint-url {S3_ENDPOINT}',
    env=env,
    shell=True,
)
