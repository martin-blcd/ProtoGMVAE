import torch.optim
from utils import get_model, plot_id_history, plot_training_curves, ImbalancedMNIST, pil_gray_loader
import shutil
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, ConcatDataset
import logging
from train import Trainer, flatten_mnist, ProtoTrainer
from save import *
from config import *
import loss
import random
import dataloader_qd as dl


makedir(output_dir)
log = logging.getLogger(__name__)
logging.basicConfig(filename=output_dir + 'main.log', encoding='utf-8', level=logging.INFO)
np.random.seed(205)
torch.manual_seed(205)

def main():
    """_summary_

    Args:
        config (dict): Configuration dictionary containing experiment parameters.
    """

    log.info(f"Output directory: {output_dir}")
    print(f"Output directory: {output_dir}")

    shutil.copy(src=os.path.join(os.getcwd(), 'config.py'), dst=output_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'models.py'), dst=output_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'modules.py'), dst=output_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'loss.py'), dst=output_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'train.py'), dst=output_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'main.py'), dst=output_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'utils.py'), dst=output_dir)

    model, criterion = get_model(**model_cfg)

    # Set up data loaders
    # Define the transformation
    transform = transforms.ToTensor()
    if data_type == 'mnist':
        train_dataset = datasets.MNIST(root='data', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root='data', train=False, transform=transform, download=True)
        if imbalanced:
            ratio = np.random.rand(10)
            train_dataset = ImbalancedMNIST(root='data', train=True, transform=transform, download=True,
                                            imbalance_ratio=ratio)
            train_dataset_bu = ImbalancedMNIST(root='data', train=True, transform=transform, download=True,
                                            imbalance_ratio=ratio)
            test_dataset = ImbalancedMNIST(root='data', train=False, transform=transform, download=True,
                                           imbalance_ratio=ratio)
            test_dataset_bu = ImbalancedMNIST(root='data', train=False, transform=transform, download=True,
                                           imbalance_ratio=ratio)
        if binary_classification:
            idx = train_dataset.targets%2==0
            train_dataset.targets[~idx] = 1
            train_dataset.targets[idx] = 0
            idx = test_dataset.targets%2==0
            test_dataset.targets[~idx] = 1
            test_dataset.targets[idx] = 0
        rgb = False

    elif data_type == 'fmnist':
        mean = 0.5
        var = 0.5
        transform = transforms.Compose([transforms.ToTensor(),
            #transforms.Normalize(mean,var)
            ])
        train_dataset = datasets.FashionMNIST(root='data', train=True, transform=transform, download=True)
        test_dataset = datasets.FashionMNIST(root='data', train=False, transform=transform, download=True)
        rgb = False
		    
    elif data_type == 'svhn':
        mean = [0.4377, 0.4438, 0.4728]
        var=[0.198, 0.201, 0.197]
        transform = transforms.Compose([transforms.ToTensor(),
            transforms.ColorJitter(0.2,0.2,0.2)
            #transforms.Normalize(mean,var)
            ])
        train_dataset = datasets.SVHN(root='data', split='train', transform=transform, download=True)
        if extra_data:
            extra_dataset = datasets.SVHN(root='data', split='extra', transform=transform, download=True)
            train_dataset = ConcatDataset([train_dataset, extra_dataset])
        test_dataset = datasets.SVHN(root='data', split='test', transform=transform, download=True)
        rgb = True

    elif data_type == 'cifar':
        #mean = [0.4914, 0.4822, 0.4465]
        #var = [0.247, 0.243, 0.261]
        mean = [0.5,0.5,0.5]
        var = [0.5,0.5,0.5]
        transform = transforms.Compose([
            #transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(p=0.5),
            #transforms.RandomRotation(10),
            transforms.ColorJitter(),
            transforms.RandomCrop(32, padding=2),
            #transforms.Resize((64,64)),
            transforms.ToTensor(),
            #transforms.Normalize(mean,var)
            ])
        transform_test = transforms.Compose([
            #transforms.Resize((64,64)),
            transforms.ToTensor(),
            #transforms.Normalize(mean,var),
            ])
        train_dataset = datasets.CIFAR10(root='data', train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(root='data', train=False, transform=transform_test, download=True)
        rgb = True

    elif data_type == 'quickdraw':
        train_dataset = dl.QuickDraw(ncat=model_cfg['num_classes'], mode='train', root_dir='data/quickdraw/')
        test_dataset = dl.QuickDraw(ncat=model_cfg['num_classes'], mode='test', root_dir='data/quickdraw/')

    elif data_type == 'tnfa100':
        transform = transforms.Compose([
            #transforms.Grayscale(),
            #Laplacian(),
            transforms.Resize(64),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomHorizontalFlip(0.5),
            #transforms.CenterCrop(32),
            transforms.ToTensor(),
            #transforms.Normalize(0.1762,0.1362),
            ])
        transform_test = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            #transforms.Normalize(0.1762,0.1362),
            ])
        train_dataset = datasets.ImageFolder('data/TNFA100_fused_sorted_2/train/',loader=pil_gray_loader, transform=transform)
        test_dataset = datasets.ImageFolder('data/TNFA100_fused_sorted_2/test/',loader=pil_gray_loader, transform=transform_test)
        rgb = False
        
    elif data_type == 'tnfa50':
        transform = transforms.Compose([
            #transforms.Grayscale(),
            #Laplacian(),
            transforms.Resize(64),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomHorizontalFlip(0.5),
            #transforms.CenterCrop(32),
            transforms.ToTensor(),
            #transforms.Normalize(0.1762,0.1362),
            ])
        transform_test = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            #transforms.Normalize(0.1762,0.1362),
            ])
        train_dataset = datasets.ImageFolder('data/TNFA50_fused/train/',loader=pil_gray_loader, transform=transform)
        test_dataset = datasets.ImageFolder('data/TNFA50_fused/test/',loader=pil_gray_loader, transform=transform_test)
        rgb = False

    train_loader = DataLoader(dataset=train_dataset, **train_loader_cfg)
    test_loader = DataLoader(dataset=test_dataset, **test_loader_cfg)

    optimizer_specs = [{'params':model.qy_x.qy_logit.parameters(),'lr':optimizer_cfg['lr']},{'params':model.qz_xy.parameters(),'lr':optimizer_cfg['lr']},{'params':model.px_z.parameters(),'lr':optimizer_cfg['lr']}]
    #optimizer = torch.optim.Adam(optimizer_specs)
    optimizer = torch.optim.Adam(params=model.parameters(),lr=optimizer_cfg['lr'],amsgrad=optimizer_cfg['amsgrad'])
    #optimizer = torch.optim.SGD(params=model.parameters(),lr=optimizer_cfg['lr'])
    optimizer_class = torch.optim.Adam(params=model.classif_layer.parameters())
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=optimizer_cfg['step'], gamma=optimizer_cfg['gamma'])
    scheduler = None

    transform_fn = flatten_mnist if model_cfg['encoder_type']=='FC' else None
    
    if warmup:
        warm_criterion = loss.ProtoTotalloss(model_cfg['k'], loss.MSE(), {'classif':1, 'kl':0, 'kl_y':0, 'recons':1, 'l1':0, 'feat_classif':0})
        warm_trainer = ProtoTrainer(
            model=model,
            optimizer=optimizer,
            criterion=warm_criterion,
            train_loader=train_loader,
            test_loader=test_loader,
            transform_fn=transform_fn,
            path=output_dir,
            scheduler=scheduler,
            **init_params)
        
        log.info(f"Warmup started. Epochs: {warmup_epochs}")
        warm_trainer.train(warmup_epochs)
        log.info("Warmup finished.")
        
        if freeze_after_warmup:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classif_layer.parameters():
                param.requires_grad = True
    
    trainer = ProtoTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        test_loader=test_loader,
        transform_fn=transform_fn,
        path=output_dir,
        scheduler=scheduler,
        **init_params)

    log.info(f"Training started. Epochs: {epochs}")
    trainer.train(epochs)
    log.info("Training finished.")

    history = trainer.history
    ids_history = trainer.ids_history
    
    if data_type == 'mnist':
        if imbalanced:
            train_loader = DataLoader(dataset=train_dataset_bu, **train_loader_cfg)
            test_loader = DataLoader(dataset=test_dataset_bu, **test_loader_cfg)

    save_model(model, output_dir, exp_name)
    plot_training_curves(history, output_dir)
    save_projections(model, test_loader, output_dir+'test_')
    save_projections(model, train_loader, output_dir+'train')
    save_images(model.visualize_all_components(), output_dir,rgb=rgb)
    #plot_id_history(ids_history, output_dir, model_cfg['input_size'], model_cfg['image_channels'], model_cfg['k'], model_cfg['encoder_type'])

if __name__ == "__main__":
    main()
