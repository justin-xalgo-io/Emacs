aws iam create-user --user-name justin
aws iam attach-user-policy --user-name justin --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam create-access-key --user-name justin
