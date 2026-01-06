


def action_recognition(image, clip_action_bag, clip_reid_bag, config):
    # resize frames to shortside
    w, h, _ = image.shape
    short_side = min(w, h)
    new_w, new_h = mmcv.rescale_size((w, h), (short_side, np.Inf))
    # frames = [mmcv.imresize(self.image, (new_w, new_h)) for img in self.image]
    w_ratio, h_ratio = new_w / w, new_h / h

    human_detections = self.clip_action_bag
    for i in range(len(human_detections)):
        det = human_detections[i].to('cuda')
        det[:, 0:4:2] *= w_ratio
        det[:, 1:4:2] *= h_ratio
        human_detections[i] = det[:, :4]

    img_norm_cfg = dict(
        mean=np.array(self.config.model.data_preprocessor.mean),
        std=np.array(self.config.model.data_preprocessor.std),
        to_rgb=False)

    prediction_out, proposal_out, reid_out = [], [], []

    for (proposal, reid) in zip(human_detections, self.clip_reid_bag):
        if proposal.shape[0] == 0:
            continue

        _ = [mmcv.imnormalize_(img, **img_norm_cfg) for img in self.clip]
        # THWC -> CTHW -> 1CTHW
        input_array = np.stack(self.clip).transpose((3, 0, 1, 2))[np.newaxis]
        input_tensor = torch.from_numpy(input_array).to('cuda')


        # input_tensor is a tensor of shape (1, 3, 8, 2160, 3840)
        # resize input_tensor to 256x256, and also resize proposal
        # input_tensor = torch.nn.functional.interpolate(input_tensor.squeeze(0), size=(256, 256), mode='bilinear', align_corners=False).unsqueeze(0)
        # proposal = proposal * torch.tensor([256.0 / new_w, 256.0 / new_h, 256.0 / new_w, 256.0 / new_h]).to(proposal.device)

        datasample = ActionDataSample()
        datasample.proposals = InstanceData(bboxes=proposal)
        datasample.set_metainfo(dict(img_shape=(new_h, new_w)))
        with torch.no_grad():
            result = self.action_recognizer(input_tensor, [datasample], mode='predict')
            scores = result[0].pred_instances.scores
            prediction = []
            # N proposals
            for i in range(proposal.shape[0]):
                prediction.append([])
            # Perform action score thr
            for i in range(scores.shape[1]):
                if i not in self.label_map:
                    continue
                for j in range(proposal.shape[0]):
                    if scores[j, i] > args.action_score_thr:
                        try:
                            action = self.label_map[i]
                        except:
                            action = 'unknown'
                        prediction[j].append((action, scores[j, i].item()))

            self.action_label = prediction
            self.action_proposal = proposal
            self.action_reid = reid
            # the first valid prediction of each clip is enough for analysis
            if sum(len(act) for act in prediction) > 0:
                return