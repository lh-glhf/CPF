"""train scripts"""
import os
from config import parse_args
from torchseq.model_factory import create_model


def train(args):
    exp = create_model(model_name=args.model, data_name=args.data, pretrained=not args.do_train, checkpoint_path=args.ckpt_path, **vars(args))
    if args.model in ['UniTS_pretrain', 'UniTS_sup']:
        if args.do_train or args.ckpt_path == '':
            if not os.path.exists("./checkpoints/train_ckpt"):
                os.mkdir("./checkpoints/train_ckpt")
            for i in range(args.itr):
                exp.train(i)
        else:
            exp.test(0)
    if args.do_train or args.ckpt_path == '':
        if not os.path.exists("./checkpoints/train_ckpt"):
            os.mkdir("./checkpoints/train_ckpt")
        for i in range(args.itr):
            exp.train(i)
            exp.test(i)
    else:
        exp.test(0)


if __name__ == "__main__":
    my_args = parse_args()
    train(my_args)
