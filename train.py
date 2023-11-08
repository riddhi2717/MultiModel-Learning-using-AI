import os
import sys
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from collections import OrderedDict
from mypath import Path
from dataset_arch_search import ClassificationDataset
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from utils.utils import AverageMeter
from utils.utils import accuracy
from auto_deeplab import AutoDeeplab, AutoDeeplabParallel
from decoding_formulas import Decoder
from config_utils.search_args import obtain_search_args
from utils.copy_state_dict import copy_state_dict
import apex
try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False


print('working with pytorch version {}'.format(torch.__version__))
print('with cuda version {}'.format(torch.version.cuda))
print('cudnn enabled: {}'.format(torch.backends.cudnn.enabled))
print('cudnn version: {}'.format(torch.backends.cudnn.version()))

torch.backends.cudnn.benchmark = True

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        self.use_amp = True if (APEX_AVAILABLE and args.use_amp) else False
        self.opt_level = args.opt_level

        kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last':True}
        
        path_to_file = os.path.join("dataset_files", args.dataset)
        path1 = os.path.join(path_to_file, "train.txt")
        path3 = os.path.join(path_to_file, "validation.txt")
        train_data = ClassificationDataset(args, args.train_split, train=True,arch_split_file=path1)
        
        val_data = ClassificationDataset(args, args.train_split, train=True,arch_split_file=path3)


        self.train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            pin_memory=True, num_workers=4)
        
       

        self.val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=args.batch_size,
            pin_memory=True, num_workers=4)
        
        self.test_loader = None
        self.nclass=args.no_of_classes
        
#         self.train_loaderA, self.train_loaderB, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                raise NotImplementedError
                #if so, which trainloader to use?
                # weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
            
        # previous
        # self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.criterion = nn.CrossEntropyLoss().cuda()
        
        # Define network
        if args.model_parallel:
            model = AutoDeeplabParallel (self.nclass, self.args.layers, self.args.channels, self.args.parallel_threshold, 
                                         self.criterion, self.args.filter_multiplier,self.args.steps, self.args.step)
        else:
            model = AutoDeeplab (self.nclass, self.args.layers, self.args.channels, self.criterion, self.args.filter_multiplier,
                                 self.args.steps, self.args.step)
        optimizer = torch.optim.SGD(
            model.weight_parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )

        self.model, self.optimizer = model, optimizer

        self.architect_optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                                    lr=args.arch_lr, betas=(0.9, 0.999),
                                                    weight_decay=args.arch_weight_decay)

        # Define Evaluator
        # self.evaluator = Evaluator(self.nclass)
        
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.trainA_loader), min_lr=args.min_lr)
        # TODO: Figure out if len(self.train_loader) should be devided by two ? in other module as well
        # Using cuda
        if args.cuda:
            self.model = self.model.cuda()


        # mixed precision
        if self.use_amp and args.cuda:
            keep_batchnorm_fp32 = True if (self.opt_level == 'O2' or self.opt_level == 'O3') else None

            # fix for current pytorch version with opt_level 'O1'
            if self.opt_level == 'O1' and torch.__version__ < '1.3':
                for module in self.model.modules():
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        # Hack to fix BN fprop without affine transformation
                        if module.weight is None:
                            module.weight = torch.nn.Parameter(
                                torch.ones(module.running_var.shape, dtype=module.running_var.dtype,
                                           device=module.running_var.device), requires_grad=False)
                        if module.bias is None:
                            module.bias = torch.nn.Parameter(
                                torch.zeros(module.running_var.shape, dtype=module.running_var.dtype,
                                            device=module.running_var.device), requires_grad=False)

            # print(keep_batchnorm_fp32)
            self.model, [self.optimizer, self.architect_optimizer] = amp.initialize(
                self.model, [self.optimizer, self.architect_optimizer], opt_level=self.opt_level,
                keep_batchnorm_fp32=keep_batchnorm_fp32, loss_scale="dynamic")

            print('cuda finished')


        # Using data parallel
        if args.cuda and len(args.gpu_ids)>1:
            if self.opt_level == 'O2' or self.opt_level == 'O3':
                print('currently cannot run with nn.DataParallel and optimization level', self.opt_level)
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            print('training on multiple-GPUs')


        # Resuming checkpoint
        self.start_epoch = 0
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            self.start_epoch = checkpoint['epoch']

            print("che_epoch",self.start_epoch)
            # if the weights are wrapped in module object we have to clean it
            if args.clean_module:
                self.model.load_state_dict(checkpoint['state_dict'])
                state_dict = checkpoint['state_dict']
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove 'module.' of dataparallel
                    new_state_dict[name] = v
                # self.model.load_state_dict(new_state_dict)
                copy_state_dict(self.model.state_dict(), new_state_dict)

            else:
                if torch.cuda.device_count() > 1 or args.load_parallel:
                    # self.model.module.load_state_dict(checkpoint['state_dict'])
                    copy_state_dict(self.model.module.state_dict(), checkpoint['state_dict'])
                else:
                    # self.model.load_state_dict(checkpoint['state_dict'])
                    copy_state_dict(self.model.state_dict(), checkpoint['state_dict'])


            if not args.ft:
                # self.optimizer.load_state_dict(checkpoint['optimizer'])
                copy_state_dict(self.optimizer.state_dict(), checkpoint['optimizer'])
            # print(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            
            args.start_epoch = 0

    def training(self, epoch):
        losses = AverageMeter()
        acc = AverageMeter()
        
        self.model.train()
        tbar = tqdm(self.trainA_loader)
        
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            # print(epoch)
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)
            if self.use_amp:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()

            if epoch >= self.args.alpha_epoch:
                search = next(iter(self.trainB_loader))
                image_search, target_search = search['image'], search['label']
                if self.args.cuda:
                    image_search, target_search = image_search.cuda (), target_search.cuda()

                self.architect_optimizer.zero_grad()
                output_search = self.model(image_search)
                arch_loss = self.criterion(output_search, target_search)
                if self.use_amp:
                    with amp.scale_loss(arch_loss, self.architect_optimizer) as arch_scaled_loss:
                        arch_scaled_loss.backward()
                else:
                    arch_loss.backward()
                self.architect_optimizer.step()

            
            k = 5 if self.args.no_of_classes>=5 else self.args.no_of_classes
            acc1, acc5 = accuracy(output, target, topk=(1, k))
            n = image.size(0)
            acc.update(acc1.item(), n)
            losses.update(loss.item(), n)
            
            tbar.set_description('Train loss: %.3f' % (losses.avg))

        return acc.avg, losses.avg


    def validation(self, epoch):
        losses = AverageMeter()
        acc = AverageMeter()
        
        self.model.eval()
        tbar = tqdm(self.val_loader, desc='\r')

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            
            k = 5 if self.args.no_of_classes>=5 else self.args.no_of_classes
            acc1, acc5 = accuracy(output, target, topk=(1, k))
            loss = self.criterion(output, target)
            n = image.size(0)
            acc.update(acc1.item(), n)
            losses.update(loss.item(), n)
            
            tbar.set_description('Test loss: %.3f' % losses.avg)

        return acc.avg, losses.avg
    
            
def main():
    args = obtain_search_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 40,
            'pascal': 50,
            'kd':10
        }
        args.epochs = epoches[args.dataset.lower()]

    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    best_acc = 0.0
    for epoch in range(trainer.start_epoch, trainer.args.epochs):
        # print(epoch)
        train_acc, train_loss = trainer.training(epoch)
        
        trainer.writer.add_scalar('train/loss_epoch', train_loss, epoch)
        trainer.writer.add_scalar('train/acc_epoch', train_acc, epoch)
        print(f"TRAIN Epoch: {epoch} acc: {train_acc:.2f} loss: {train_loss:.2f}")
        
        if len(args.gpu_ids) > 1:
            [alp, bet, gam] = trainer.model.module.arch_parameters()
        else:
            [alp, bet, gam] = trainer.model.arch_parameters()
        print(gam)
        decoder = Decoder(alphas=alp, gammas=gam, betas=bet, steps=trainer.args.steps)
        network_path, network_path_space = decoder.viterbi_decode()
        trainer.writer.add_text('network_path',str(network_path),epoch)
        print("Network Path:", network_path)
        
        valid_acc, valid_loss = trainer.validation(epoch)
        
        trainer.writer.add_scalar('valid/loss_epoch', valid_loss, epoch)
        trainer.writer.add_scalar('valid/acc_epoch', valid_acc, epoch)
        print(f"VALID Epoch: {epoch} acc: {valid_acc:.2f} loss: {valid_loss:.2f}")
        
        new_acc = valid_acc
        is_best = False
        if new_acc >= best_acc:
            is_best = True
            best_acc = new_acc
            
        if len(args.gpu_ids) > 1:
            state_dict = trainer.model.module.state_dict()
        else:
            state_dict = trainer.model.state_dict()
        trainer.saver.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': state_dict,
            'best_pred' : best_acc,
            'optimizer': trainer.optimizer.state_dict(),
        }, is_best)
        
        if epoch == trainer.args.epochs - 1:
            save_arch_structures(trainer)


    trainer.writer.close()


def save_arch_structures(trainer):
    
    #[alp, bet] = trainer.model.module.arch_parameters()
    checkpoint = torch.load(os.path.join(trainer.saver.experiment_dir,"model_best.pth.tar"))
    alp = checkpoint['state_dict']['alphas']
    gam = checkpoint['state_dict']['gammas']
    bet = checkpoint['state_dict']['betas']
    decoder = Decoder(alphas=alp, gammas = gam, betas=bet, steps=trainer.args.steps)
    network_path, network_path_space = decoder.viterbi_decode()
    genotype = decoder.genotype_decode()

    print('Final Network Arch:', network_path)
    print('Final Cell Arch:\n', genotype)

    dir_name = trainer.saver.experiment_dir
    network_path_filename = os.path.join(dir_name, 'network_path')
    network_path_space_filename = os.path.join(dir_name, 'network_path_space')
    genotype_filename = os.path.join(dir_name, 'genotype')
    
    np.save(network_path_filename, network_path)
    np.save(network_path_space_filename, network_path_space)
    np.save(genotype_filename, genotype)
    
if __name__ == "__main__":
   main()
