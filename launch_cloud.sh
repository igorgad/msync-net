#!/usr/bin/env bash

JOB_DIR=gs://ms-bucket/MSYNC/workspace
DATASET_FILE=gs://ms-bucket/datasets/MedleyDB/MSYNC-MedleyDB_v2.tfrecord
DATASET_AUDIO_DIR=gs://ms-bucket/datasets/MedleyDB/Audio
LOG_DIR=gs://ms-bucket/MSYNC/logs
EPOCHS=50

JOB_NAME=train_master_lstm_singlesoftmax_job0

gcloud ml-engine jobs submit training $JOB_NAME \
                                    --stream-logs \
                                    --runtime-version 1.12 \
                                    --python-version 3.5 \
                                    --job-dir $JOB_DIR \
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
                                    --verbose 2  \
