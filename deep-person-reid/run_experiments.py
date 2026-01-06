import os
import csv
import time
from datetime import datetime

import torch
import torchreid


MODELS = [
    # 'resnet18', 'resnet50', 'resnet101', 'resnet152',
    # 'resnext50_32x4d', 'resnext101_32x8d',
    # 'se_resnet50', 'se_resnet50_fc512', 'se_resnet101',
    # 'se_resnext50_32x4d', 'se_resnext101_32x4d',
    # 'densenet121', 'densenet169', 'densenet201', 'densenet161',
    # 'inceptionresnetv2', 'xception', 'nasnsetmobile',
    # 'mobilenetv2_x1_0', 'mobilenetv2_x1_4', 'shufflenet',
    # 'squeezenet1_0', 'squeezenet1_1',
    # 'shufflenet_v2_x0_5',
    # 'shufflenet_v2_x2_0', 'mudeep'
    'pcb_p6', 'pcb_p4', 'mlfn', 'osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5',
    'osnet_x0_25', 'osnet_ibn_x1_0', 'osnet_ain_x1_0', 'osnet_ain_x0_75',
    'osnet_ain_x0_5', 'osnet_ain_x0_25'
]

MAX_EPOCHS = [10, 20, 30, 40]
REPEATS = 2


def ensure_log_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def get_log_writer(csv_path):
    """Create CSV writer and write header if file is new."""
    file_exists = os.path.isfile(csv_path)
    f = open(csv_path, mode='a', newline='')
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow([
            'timestamp', 'model', 'epoch_setting', 'repeat', 'dataset',
            'mAP', 'rank1', 'rank5', 'rank10', 'rank20',
            'speed_sec_per_batch', 'time_elapsed_min',
            'save_dir'
        ])
        f.flush()
    return f, writer


def train_and_evaluate(model_name, max_epoch, repeat_idx, csv_writer, csv_file):
    # Data manager mirrors demo.py settings
    datamanager = torchreid.data.ImageDataManager(
        root='path_to_your_root',
        sources='ours',
        height=256,
        width=128,
        batch_size_train=32,
        batch_size_test=100,
        transforms=["random_flip", "random_crop"],
    )

    model = torchreid.models.build_model(
        name=model_name,
        num_classes=datamanager.num_train_pids,
        loss='softmax',
        pretrained=True,
    )

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torchreid.optim.build_optimizer(model, optim='adam', lr=3e-4)
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer, lr_scheduler='single_step', stepsize=20
    )

    engine = torchreid.engine.ImageSoftmaxEngine(datamanager, model, optimizer, scheduler=scheduler)

    save_dir = os.path.join('log', f'{model_name}-softmax-our-e{max_epoch}-r{repeat_idx+1}')
    os.makedirs(save_dir, exist_ok=True)

    # Train and automatically run final test (per engine.run)
    engine.run(
        max_epoch=max_epoch,
        save_dir=save_dir,
        print_freq=10,
    )

    # Extract metrics stored during the last evaluation
    results = getattr(engine, 'last_eval_results', None)
    if not results:
        # Fallback: explicitly run test to populate
        engine.test()
        results = getattr(engine, 'last_eval_results', None)

    # Prepare metrics for logging
    timestamp = datetime.utcnow().isoformat()
    dataset = results.get('dataset_name', 'unknown') if results else 'unknown'
    mAP = results.get('mAP', None) if results else None
    cmc = results.get('cmc', []) if results else []
    rank1 = cmc[0] if len(cmc) > 0 else None
    rank5 = cmc[4] if len(cmc) > 4 else None
    rank10 = cmc[9] if len(cmc) > 9 else None
    rank20 = cmc[19] if len(cmc) > 19 else None
    speed = results.get('speed_sec_per_batch', None) if results else None
    elapsed_min = results.get('time_elapsed_min', None) if results else None

    csv_writer.writerow([
        timestamp, model_name, max_epoch, repeat_idx + 1, dataset,
        mAP, rank1, rank5, rank10, rank20,
        speed, elapsed_min,
        save_dir,
    ])
    csv_file.flush()


def main():
    ensure_log_dir('log')
    csv_path = os.path.join('log', 'experiments.csv')
    csv_file, csv_writer = get_log_writer(csv_path)

    try:
        for model_name in MODELS:
            for max_epoch in MAX_EPOCHS:
                for repeat_idx in range(REPEATS):
                    train_and_evaluate(model_name, max_epoch, repeat_idx, csv_writer, csv_file)
    finally:
        csv_file.close()


if __name__ == '__main__':
    main()


