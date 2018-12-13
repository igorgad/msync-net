#!/usr/bin/env bash

DATASET_FILE=gs://msync-bucket/datasets/BACH10/MSYNC-bach10.tfrecord
DATASET_AUDIO_DIR=gs://msync-bucket/datasets/BACH10/Audio/
LOG_DIR=gs://msync-bucket/logs/MSYNC/
EPOCHS=50

JOB_NAME=msyncteste3

gcloud ml-engine jobs submit training $JOB_NAME \
                                    --stream-logs \
                                    --runtime-version 1.10 \
                                    --python-version 3.5 \
                                    --job-dir $LOG_DIR \
                                    --package-path trainer \
                                    --module-name trainer.msync \
                                    --region europe-west1 \
                                    --scale-tier basic \
                                    -- \
                                    --dataset_file $DATASET_FILE \
                                    --dataset_audio_dir $DATASET_AUDIO_DIR \
                                    --logdir $LOG_DIR \
                                    --epochs $EPOCHS
