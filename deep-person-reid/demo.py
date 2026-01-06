import torchreid



def main():

    datamanager = torchreid.data.ImageDataManager(
                root='path_to_your_root',
                sources='ours',
                height=256,
                width=128,
                batch_size_train=32,
                batch_size_test=100,
                transforms=["random_flip", "random_crop"]
            )
    # path = '/home/wzhangbu/elderlycare/weights/osnet_x1_0_market1501.pt'
    path = '/home/wzhangbu/deep-person-reid/weights/osnet_x1_0_market1501.pth'
    '''
        osnet_x1_0 /home/wzhangbu/deep-person-reid/weights/osnet_x1_0_market1501.pth
        resnet50_fc512 /home/wzhangbu/deep-person-reid/weights/resnet50_fc512_market_xent.pth.tar
        mobilenetv2_x1_4 /home/wzhangbu/deep-person-reid/weights/mobilenetv2_1dot4_market.pth.tar
        mlfn /home/wzhangbu/deep-person-reid/weights/mlfn_market_xent.pth.tar
    '''
    model = torchreid.models.build_model(
                name='pcb_p6',
                num_classes=datamanager.num_train_pids,
                loss='softmax',
                pretrained=True
            )

    model = model.cuda()
    optimizer = torchreid.optim.build_optimizer(
        model, optim='adam', lr=3e-4
    )
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler='single_step',
        stepsize=20
    )
    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager, model, optimizer, scheduler=scheduler
    )
    engine.run(
        max_epoch=30,
        save_dir='log/osnet_x1_0-softmax-our',
        print_freq=10,
        # test_only=True
    )


if __name__ == '__main__':
    main()
