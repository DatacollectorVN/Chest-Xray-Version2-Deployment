from src.utils import check_connect_rds, check_connect_s3
INI_FILE_PATH = 'IAC/credential_aws.ini'
SECTION_RDS = 'Credential-AWS-RDS-MySQL'
SECTION_S3 = 'Credential-AWS-S3'
if __name__ == '__main__':
    check_connect_rds(INI_FILE_PATH, SECTION_RDS)
    check_connect_s3(INI_FILE_PATH, SECTION_S3)