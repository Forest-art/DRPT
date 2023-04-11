from torch.nn.modules.loss import CrossEntropyLoss


def loss_calu(predict, target, config, dataset):
    loss_fn = CrossEntropyLoss()
    batch_img, batch_attr, batch_obj, batch_target = target
    batch_attr = batch_attr.cuda()
    batch_obj = batch_obj.cuda()
    batch_target = batch_target.cuda()

    logits = predict
    loss = loss_fn(logits, batch_target)
    return loss 