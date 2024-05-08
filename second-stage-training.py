from __future__ import print_function, division
import sys, argparse
import torch
print(torch.__version__)

from utils.network_arch_resnet import *
from utils.trainval import *
from utils.eval_funcs import print_accuracy
from utils.class_balanced_loss import CB_loss
from utils import dataloader
from utils.resnet_cifar import resnet32
import warnings # ignore warnings
warnings.filterwarnings("ignore")
print(sys.version)
print(torch.__version__)

seed = 0
#Setup config parameters
torch.manual_seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser()

# Hardware
parser.add_argument("--device_id", type=str, default="0", help="")

# Directories
parser.add_argument("--dataset", type=str, default='iNaturalist18', help="dataset name")

# Hyper parameters
parser.add_argument("--bs", type=int, default=64, help="Mini batch size")  #imagenet512
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for training")  #imagenet 0.01
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers in dataloader")
parser.add_argument("--weight_decay", type=float, default=0, help="weight decay of the optimizer")  #imagenet 0
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum of SGD function")
parser.add_argument("--num_classes", type=int, default=8142, help="number of classes")
parser.add_argument("--imb_factor", type=float, default=0.01, help="level of imbalance")
parser.add_argument("--encoder_layers", type=int, default=50, help="model size of encoder")
parser.add_argument("--imb_type", type=str, default='exp', help="imblance type")
parser.add_argument("--print_fr", type=int, default=1, help="print frequency")
parser.add_argument("--num_epochs", type=int, default=200, help="Numbe of epochs to train")

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id

data_root = {'ImageNet': './datasets/ImageNet',
             'Places': './datasets/Places',
             'iNaturalist18': './datasets/iNaturalist18',
             'CIFAR10': './datasets/cifar10',
             'CIFAR100': './datasets/cifar100',
             }

# set device, which gpu to use.
device ='cpu'
if torch.cuda.is_available(): 
    device='cuda'

curr_working_dir = os.getcwd()
project_name = 'stage_2'
isPretrained = False

torch.cuda.device_count()
torch.cuda.empty_cache()
save_dir = path.join(curr_working_dir, 'exp', project_name)
if not os.path.exists(save_dir): os.makedirs(save_dir)
log_filename = os.path.join(save_dir, 'train.log')

#prepare dataset

phase = ['train', 'test']
dataset = {x: dataloader.load_data(data_root=data_root[args.dataset.rstrip('_LT')],
                    dataset=args.dataset, phase=x,
                    cifar_imb_ratio=args.imb_factor if x == 'train' else 1,
                    shuffle=x == 'train')
        for x in phase}
dataloaders = {x: DataLoader(dataset=dataset[x], batch_size=args.bs,
                      shuffle=x == 'train', num_workers=args.num_workers, pin_memory=True)
               for x in phase}

img_num_per_cls, new_labelList = dataset['train'].img_num_per_cls, dataset['train'].targets

print('#train batch:', len(dataloaders['train']), '\t#test batch:', len(dataloaders['test']))

path = 'naive_model' + '_' + args.dataset + '_best.paramOnly'
################## load base model ###################
path_to_model = os.path.join('./exp/stage_1', path)

base_model = ResnetEncoder(args.encoder_layers, isPretrained, embDimension=args.num_classes, poolSize=4)
# base_model = resnet32(100, return_features=True)
base_model = nn.DataParallel(base_model).cuda()
base_model.load_state_dict(torch.load(path_to_model, map_location=device))
base_model.to(device)

models = {'base': base_model}
model = copy.deepcopy(base_model)
model_name = 'IWB_' + args.dataset

def CB_lossFunc(logits, labelList): #defince CB loss function
    return CB_loss(labelList, logits, img_num_per_cls, args.num_classes, "softmax", 0.9999, 2, device)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, eta_min=0.0)

save_dir = os.path.join(curr_working_dir, 'exp', project_name)

trackRecords = train_model_stage2(dataloaders, model, CB_lossFunc, optimizer, scheduler, pgdFunc=None,
                           num_epochs=args.num_epochs, device=device, work_dir='./exp/'+project_name,
                           model_name=model_name, labelnames=new_labelList, img_num_per_class=img_num_per_cls, args=args)

# load model with best epoch accuracy
path_to_clsnet = os.path.join(save_dir, model_name+'_best.paramOnly')
model.load_state_dict(torch.load(path_to_clsnet, map_location=device))

model.to(device)
model.eval()
models['Inverse weight-balancing'] = model

print_accuracy(model, dataloaders, np.array(new_labelList), img_num_per_cls=img_num_per_cls, device=device)