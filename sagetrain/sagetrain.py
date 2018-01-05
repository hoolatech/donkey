import os
import sagemaker as sage
from sagemaker import get_execution_role

sagemaker_session = sage.Session()

#role = get_execution_role()

role='arn:aws:iam::271275436050:role/service-role/AmazonSageMaker-ExecutionRole-20171206T112881'

inputs = "s3://sagemaker-us-east-1-271275436050/data/racecar/"
image="271275436050.dkr.ecr.us-east-1.amazonaws.com/deepimage"
algo = sage.estimator.Estimator(image,
                       role, 1, 'ml.p2.xlarge',
                       output_path="s3://{}/output".format("sagemaker-keji"),
                       sagemaker_session=sagemaker_session)

algo.fit(inputs)