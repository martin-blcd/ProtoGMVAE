exp_name = "cifar_rgb_k50_l5e-4_hs64_ls32_batch128_qy_eps1e-4_flips_300ep_sigma0.1_cls10"
output_dir = "./outputs/" + exp_name + "/"

data_type = 'cifar'
if data_type == 'mnist' or data_type == 'fmnist':
    #MNIST
    train_loader_cfg = {'batch_size': 128, 'shuffle' : True, 'num_workers' : 1}
    test_loader_cfg = {'batch_size': 2048, 'shuffle' : False, 'num_workers' : 1}
    binary_classification = False
    imbalanced = False
elif data_type == 'svhn' or data_type == 'cifar':
    #SVHN, CIFAR10
    train_loader_cfg = {'batch_size': 128, 'shuffle' : True, 'num_workers' : 1}
    test_loader_cfg = {'batch_size': 2048, 'shuffle' : False, 'num_workers' : 1}
    extra_data = False
else:
    train_loader_cfg = {'batch_size': 100, 'shuffle' : True, 'num_workers' : 1}
    test_loader_cfg = {'batch_size': 2048, 'shuffle' : False, 'num_workers' : 1}

#GMVAE
model_cfg = {
'k' : 50,
# 'num_classes' : 10,
'encoder_type' : "RGB",
'input_size' : 28 if data_type =='mnist' or data_type == 'fmnist' else 64 if data_type == 'tnfa100' or data_type == 'tnfa50' else 32, #careful to change it when going back to full scale analysis
#'input_size' : 32*32*3,
'hidden_size' : 64, #512
'latent_dim' : 32,
'image_channels': 1 if data_type =='mnist' or data_type == 'fmnist' or data_type == 'tnfa100' or data_type == 'tnfa50' else 3,
'recon_loss_type' : "MSE",
'eps' : 1e-6,
'model_name' : "ProtoGMVAE",
'loss_name' : "ProtoLoss",
'encoder_kwargs' : {'dropout': 0.0},
'decoder_kwargs' : {'return_probs': False, 'dropout':0.0},
'num_classes' : 2 if data_type == 'tnfa100' else 10,
'coefs' : {'classif': 10, 'kl':1, 'kl_y':1, 'recons':1, 'l1':0, 'feat_classif':0},
#'local_patches' : False,
#'overlap' : 0.875, #1 = no overlap
#'init_img_size' : 28,
#'patches_per_batch' : 32 #train_loader_cfg['batch_size']
#'cls_dropout': 0.0
}

#ADAM
optimizer_cfg = {
'_target_' : "torch.optim.adam.Adam",
'lr' : 5e-4,
'weight_decay' : 0.0,
'step' : 300,
'gamma' : 1.0,
'amsgrad' : True
}

#TRAINER
init_params = {
  'device': "cuda",
  'track_ids': False,
  'tracked_ids': {},
  'n': 1,
  'binarize_x': False
  }
epochs = 300
warmup = False
warmup_epochs = 20
freeze_after_warmup = False

