import os, json
import datetime
import argparse


class Options:
    def __init__(self, model_name: str):
        models = ["cloudnet", "cdnetv2", "hrcloudnet", "mscff", "swinunet", "mcdnet", "rdunet", "cloudmamba"]
        assert model_name in models
        parser = argparse.ArgumentParser('Cloud Detection -- air-cd')
        # params of data storage
        parser.add_argument("--root", type=str, default="./data/air-cd", help="absolute path of the dataset")
        parser.add_argument("--cloudy", type=str, default="image", help="dir name of the cloud image dataset")
        parser.add_argument("--label", type=str, default="gt", help="dir name of the label dataset")
        # params of dataset
        parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")
        parser.add_argument("--in_channels", type=int, default=4, help="number of image channels (R/G/B/NIR)")
        parser.add_argument("--num_classes", type=int, default=2, help="number of classes (0=clear 1=cloud)")
        parser.add_argument("--file_suffix", type=str, default='.tif', help="the filename suffix of train data")
        # params of model training
        parser.add_argument("--start_epoch", type=int, default=1, help="start epoch")
        parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
        parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
        parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
        parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
        parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
        # other
        parser.add_argument("--checkpoint", type=str, default='0', help="checkpoint to load pretrained models")
        parser.add_argument("--model_name", type=str, default=model_name, help="model name")
        parser.add_argument("--save_name", type=str, default=model_name, help="dir name to save model")
        parser.add_argument( "--sample_interval", type=int, default=1, help="epoch interval between sampling of images from model" )
        parser.add_argument( "--evaluation_interval", type=int, default=1, help="epoch interval between evaluation from model" )
        parser.add_argument("--checkpoint_interval", type=int, default=50, help="interval between model checkpoints")
        parser.add_argument("--time", type=str, default=self._time(), help="the run time")
        parser.add_argument("--gpu", type=str, default="1", help="GPU id(s) to use, e.g. '0' or '0,1'")

        self.args = parser.parse_args()

        base = os.path.join("checkpoints", self.args.save_name)
        os.makedirs(os.path.join(base, "show"),        exist_ok=True)
        os.makedirs(os.path.join(base, "saved_models"), exist_ok=True)
        os.makedirs(os.path.join(base, "args"),         exist_ok=True)
        os.makedirs(os.path.join(base, "evaluation"),   exist_ok=True)

    def parse(self, save_args=True):
        print(self.args)
        if save_args:
            self._save_args()
        return self.args
    
    def _time(self):
        now = datetime.datetime.now()
        year = str(now.year)
        month = str(now.month).zfill(2)
        day = str(now.day).zfill(2)
        hour = str(now.hour).zfill(2)
        date_str = year + month + day + hour
        return date_str
    
    def _save_args(self):
        out_path = os.path.join("checkpoints", self.args.save_name,
                                "args", f'{self.args.time}.args')
        with open(out_path, 'w') as f:
            json.dump(self.args.__dict__, f, indent=2)

