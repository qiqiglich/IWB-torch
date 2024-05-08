from __future__ import print_function, division
import sys
import argparse
from utils.trainval import *
from utils import dataloader
from utils.resnet_cifar import resnet32
import warnings  # ignore warnings

warnings.filterwarnings("ignore")
print(sys.version)
print(torch.__version__)

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
# Hardware
parser.add_argument("--device_id", type=str, default="2", help="")

# Directories
parser.add_argument("--dataset", type=str, default='CIFAR100_LT', help="dataset name")

# Hyper parameters
parser.add_argument("--bs", type=int, default=64, help="Mini batch size")
parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for training")  #naive 0.002
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers in dataloader")
parser.add_argument("--weight_decay", type=float, default=5e-3, help="weight decay of the optimizer")#naive 8e-5
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum of SGD function")
parser.add_argument("--num_classes", type=int, default=100, help="number of classes")
parser.add_argument("--encoder_layers", type=int, default=34, help="model size of encoder")
parser.add_argument("--print_fr", type=int, default=1, help="print frequency")
parser.add_argument("--num_epochs", type=int, default=200, help="Numbe of epochs to train")
parser.add_argument("--pretrain", type=str, default=False, help="pretrain model")

parser.add_argument("--imb_type", type=str, default='exp', help="imblance type")
parser.add_argument("--imb_factor", type=float, default=1, help="level of imbalance")

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id

data_root = {'ImageNet': './datasets/ImageNet',
             'Places': './datasets/Places',
             'iNaturalist18': './datasets/iNaturalist18',
             'CIFAR10': './datasets/cifar10',
             'CIFAR100': './datasets/cifar100',
             }
pretrain_path = "./resnet152-394f9c45.pth"

# set device, which gpu to use.
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

curr_working_dir = os.getcwd()
project_name = 'stage_1'

nClasses = args.num_classes  # number of classes in CIFAR100-LT with imbalance factor 100
batch_size = args.bs  # batch size
isPretrained = args.pretrain

torch.cuda.device_count()
torch.cuda.empty_cache()

save_dir = path.join(curr_working_dir, 'exp', project_name)
if not os.path.exists(save_dir): os.makedirs(save_dir)
log_filename = os.path.join(save_dir, 'train.log')

# ## prepare dataset

phase = ['train', 'test']
dataset = {x: dataloader.load_data(data_root=data_root[args.dataset.rstrip('_LT')],
                                   dataset=args.dataset, phase=x,
                                   cifar_imb_ratio=args.imb_factor if x == 'train' else 1,
                                   shuffle=x == 'train')
           for x in phase}
dataloaders = {x: DataLoader(dataset=dataset[x], batch_size=args.bs,
                             shuffle=x == 'train', num_workers=args.num_workers)
               for x in phase}

img_num_per_cls, new_labelList = dataset['train'].img_num_per_cls, dataset['train'].targets

print('#train batch:', len(dataloaders['train']), '\t#test batch:', len(dataloaders['test']))

loss_CrossEntropy = nn.CrossEntropyLoss()

models = {}

model_name = 'naive_model'
model = resnet32(100, return_features=True)

if isPretrained:
    new_state_dict = {}
    pretrain_state_dict = torch.load(pretrain_path, map_location=device)
    for k, v in pretrain_state_dict.items():
        k = "encoder." + k
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

model = nn.DataParallel(model).cuda()

optimizer = optim.SGD([{'params': model.parameters(), 'lr': args.lr}], lr=args.lr, momentum=0.9,
                      weight_decay=args.weight_decay)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, eta_min=0)

trackRecords = train_model_stage1(dataloaders, model, loss_CrossEntropy, optimizer, scheduler,
                                  num_epochs=args.num_epochs, device=device, work_dir='./exp/' + project_name,
                                  model_name=model_name, print_each=args.print_fr, dataset_name=args.dataset)

# load model with best epoch accuracy
path_to_clsnet = os.path.join(save_dir, model_name + '_' + args.dataset + '_best.paramOnly')
model.load_state_dict(torch.load(path_to_clsnet, map_location=device))

model.to(device)
model.eval()
