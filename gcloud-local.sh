

TRAINER_PACKAGE_PATH=$(pwd)
MAIN_TRAINER_MODULE="trainer.task"
PACKAGE_STAGING_PATH="gs://your/chosen/staging/path"

now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="your_name_$now"
JOB_DIR="gs://your/chosen/job/output/path"
REGION="us-east1"

gcloud ml-engine local train \
    --module-name msync-net.task \
    --package-path ../ \
    --job-dir $MODEL_DIR \
    -- \
    --train-files $TRAIN_DATA \
    --eval-files $EVAL_DATA \
    --train-steps 1000 \
    --eval-steps 100