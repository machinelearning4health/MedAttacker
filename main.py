import argparse
from torch.utils.data import DataLoader
from trainer import MedAttacker
from transformer import *
from dataset import EHRDataset
from hitanet import HiTANet
import os


parser = argparse.ArgumentParser()

parser.add_argument("-hs1", "--hidden1", type=int, default=256, help="hidden size of transformer model in first layer")
parser.add_argument("-l1", "--layers1", type=int, default=8, help="number of layers in first")
parser.add_argument("-a1", "--attn_heads1", type=int, default=8, help="number of attention heads 1")
parser.add_argument("-drop1", "--dropout1", type=float, default=0.1)

parser.add_argument("-hs2", "--hidden2", type=int, default=256, help="hidden size of transformer model in second layer")
parser.add_argument("-l2", "--layers2", type=int, default=8, help="number of layers in second")
parser.add_argument("-a2", "--attn_heads2", type=int, default=8, help="number of attention heads 2")
parser.add_argument("-drop2", "--dropout2", type=float, default=0.1)

parser.add_argument("-b", "--batch_size", type=int, default=64, help="number of batch_size")
parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
parser.add_argument("-w", "--num_workers", type=int, default=5, help="dataloader worker size")

parser.add_argument("-gpu","--with_cuda", type=int, default=0, help="training with CUDA: true, or false")

parser.add_argument("-lr","--lr", type=float, default=0.001, help="learning rate of adam")
parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

parser.add_argument("--saving_path", default='model_params/models.pth', help="Path for Saving the Models")


args = parser.parse_args()

print("hs1:{}".format(args.hidden1))
print("l1:{}".format(args.layers1))
print("hs2:{}".format(args.hidden2))
print("l2:{}".format(args.layers2))
print("lr:{}".format(args.lr))
print(args.with_cuda)


with open('./data/hf/model_inputs/hf_training_new.pickle','rb') as f:
    train_data = pickle.load(f)
train_visit = train_data[0]
train_label = train_data[1]
train_time = train_data[2]
train_dataset = EHRDataset(train_visit, train_label, train_time)

with open('./data/hf/model_inputs/hf_validation_new.pickle','rb') as f:
    validate_data = pickle.load(f)
validate_visit = validate_data[0]
validate_label = validate_data[1]
validate_time = validate_data[2]
validate_dataset = EHRDataset(validate_visit, validate_label, validate_time)

with open('./data/hf/model_inputs/hf_testing_new.pickle','rb') as f:
    test_data = pickle.load(f)
test_visit = test_data[0]
test_label = test_data[1]
test_time = test_data[2]
test_dataset = EHRDataset(test_visit, test_label, test_time)




print("Creating Dataloader")
train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=train_dataset.collate_fn,shuffle=False)
if args.test_dataset is not None:
    validate_data_loader = DataLoader(validate_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=test_dataset.collate_fn,shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=test_dataset.collate_fn,shuffle=False)
else:
    test_data_loader = None


embedding_dim = 256
code2idx_file = './data/hf/hf_code2idx_new.pickle'

with open(code2idx_file, 'rb') as f:
    code2idx = pickle.load(f)
    diagnosis_code_list = list(code2idx.keys())
    dignosis_index_list = list(code2idx.values())



validate_model = HiTANet(len(dignosis_index_list), embedding_dim, transformer_hidden = args.hidden1, attn_heads = args.attn_heads1,
                    transformer_dropout = args.dropout1, transformer_layers = args.layers1)
validate_model.load_state_dict(torch.load(args.saving_path))
validate_model.eval()

validator = MedAttacker(validate_model, train_dataloader=train_data_loader, validate_dataloader=validate_data_loader, test_dataloader=test_data_loader, with_cuda=args.with_cuda, lr=args.lr, output_dir=args.output_path)

epoch = 20
validator.attack(epoch)
