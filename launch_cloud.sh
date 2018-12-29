#!/usr/bin/env bash

DATASET_FILE=gs://msync-bucket/datasets/MedleyDB/MSYNC-MedleyDB_v2.tfrecord
DATASET_AUDIO_DIR=gs://msync-bucket/datasets/MedleyDB/Audio
LOG_DIR=gs://msync-bucket/logs/MSYNC
EPOCHS=50

JOB_NAME=notd_test21

gcloud ml-engine jobs submit training $JOB_NAME \
                                    --stream-logs \
                                    --runtime-version 1.12 \
                                    --python-version 3.5 \
                                    --job-dir $LOG_DIR \
                                    --package-path trainer \
                                    --module-name trainer.msync \
                                    --region europe-west1 \
                                    --scale-tier basic_gpu \
                                    -- \
                                    --dataset_file $DATASET_FILE \
                                    --dataset_audio_dir $DATASET_AUDIO_DIR \
                                    --logdir $LOG_DIR \
                                    --epochs $EPOCHS \
                                    --from_bucket True \
                                    --culstm True \
                                    --random_batch_size 32 \
