import torch
from models.sseg.cloudnet import CloudNet
from models.sseg.cdnetv2 import CDnetV2
from models.sseg.swinunet import SwinUnet
from models.sseg.hrcloudnet import HRcloudNet
from models.sseg.rdunet import CloudDetNet
from models.sseg.cloudmamba import CloudMambaUnet

def get_model(args, device):
    model_name = args.model_name
    in_channels, out_channels = args.in_channels, args.num_classes
    if model_name == "cloudnet":
        model = CloudNet(in_channels=in_channels, out_channels=out_channels)
    elif model_name == "cdnetv2":
        model = CDnetV2(in_channels=in_channels, out_channels=out_channels)
    elif model_name == "swinunet":
        model = SwinUnet(in_channels=in_channels, out_channels=out_channels)
    elif model_name == "hrcloudnet":
        model = HRcloudNet(in_channels=in_channels, out_channels=out_channels)
    elif model_name == "rdunet":
        model = CloudDetNet(in_channels=in_channels, out_channels=out_channels)
    elif model_name == "cloudmamba":
        model = CloudMambaUnet(in_channels=in_channels, out_channels=out_channels)
    else:
        exit("\nError: MODEL \'%s\' is not implemented!\n" % model)

    model = model.to(device)
    inputs = [torch.randn(args.batch_size, args.in_channels, args.img_size, args.img_size, device=device)]
    model.eval()
    with torch.no_grad():
        model(*inputs)
    model.train()
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("%s Params: %.2fM" % (model_name, params_num / 1e6))

    return model
