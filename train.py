import torch
import torch.nn.utils
from utils import get_model, NumpyEncoder
import torch.optim as optim
from torchvision import datasets, transforms,get_image_backend
from collections import defaultdict
import numpy as np
import json
from tqdm import tqdm
import logging
import sys
import os
from PIL import Image
from modules import EncoderFC,EncoderCONV,EncoderRGB
from save import *

log = logging.getLogger(__name__)


def flatten_mnist(tensor):
    return tensor.reshape(-1, 32*32*3)


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img


class Trainer:
    def __init__(self, model, optimizer, criterion, train_loader,
                 test_loader, device, path, track_ids=True, tracked_ids={},
                 n=1, binarize_x=False, transform_fn=flatten_mnist):
        """
        Trainer class for training and evaluating a PyTorch model.

        Args:
            model (nn.Module): The PyTorch model to train and _evaluate.
            optimizer (torch.optim.Optimizer): The optimizer for updating
                model parameters.
            criterion (callable): The loss function to compute
                the training loss.
            train_loader (torch.utils.data.DataLoader): DataLoader for
                the training dataset.
            test_loader (torch.utils.data.DataLoader): DataLoader for
                the test dataset.
            device (str): Device to run the computations on ("cuda" or "cpu").
            track_ids (bool): Flag indicating whether to track specific sample
                IDs during training (default: True).
            tracked_ids (set): Set of sample IDs to track during training
                (default: empty set).
            n (int): Number of sample IDs to track during training
                (default: 2).
            transform_fn (callable): Optional function to transform the input
            data (default: flatten_mnist).
        """
        self.optimizer = optimizer
        self.criterion = criterion

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.transform_fn = transform_fn
        self.binarize_x = binarize_x

        self.history = defaultdict(list)
        self.track_ids = track_ids
        self.tracked_ids = tracked_ids
        self.n = n
        self.ids_history = defaultdict(dict)

        self.current_epoch = 0
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and device == "cuda") else "cpu")
        self.model = model.to(self.device)
        
        self.path = path

    def train(self, epochs):
        """
        Train the model for the specified number of epochs.

        Args:
            epochs (int): Number of epochs to train the model.
        """
        log.info(f"Training on {self.device}")
        if self.track_ids:
            if len(self.tracked_ids) == 0:
                self.tracked_ids = self._get_n_ids_per_class(self.n)
            self._get_tracked_x_true()

        for epoch in range(epochs):
            self._train_epoch()
            self._evaluate()

            # Track history for ids over epochs:
            if self.track_ids:
                self._infer_tracked_ids()

            train_loss = self.history["train_loss"][-1]
            test_loss = self.history["test_loss"][-1]
            train_acc = self.history["train_accuracy"][-1]
            test_acc = self.history["test_accuracy"][-1]

            log.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f},"
                     f"Test Loss: {test_loss:.4f}, Train Acc: {train_acc:.4f} "
                     f"Test Acc: {test_acc:.4f}")
        # After trainig actions:
        os.makedirs(self.path, exist_ok=True)
        history_path = os.path.join(self.path, "history.json")
        self.dump_to_json(self.history, history_path, indent=4)
        ids_history_path = os.path.join(self.path, "ids_history.json")
        self.dump_to_json(self.ids_history, ids_history_path)

    def get_accuracy(self, y_true, y_pred):
        """
        Get accuracy.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        corrected_pred = -np.ones_like(y_pred)

        for cls in np.unique(y_pred):
            indx = y_pred == cls
            true_cls = np.bincount(y_true[indx]).argmax()
            corrected_pred[indx] = true_cls
        acc = np.mean(y_true == corrected_pred)
        return acc

    def _train_epoch(self):
        """
        Training loop
        """
        model = self.model.train()
        optimizer = self.optimizer
        criterion = self.criterion
        dataloader = self.train_loader
        device = self.device

        running_loss = 0.0
        running_entropy = 0.0
        pred_labels = []
        true_labels = []

        for data, labels in tqdm(dataloader):
            data = data.to(device)

            if self.transform_fn:
                data = self.transform_fn(data)

            if self.binarize_x:
                batch = data.shape[0]
                thresholds = torch.rand((batch, 1)).to(data.device)
                data = torch.where(data > 0.1, 1.0, 0.0)

            labels = labels.to(device)

            optimizer.zero_grad()
            out_train, out_infer = model(data)
            loss = criterion(data, out_train)
            loss['total_loss'].backward()
            optimizer.step()

            running_loss += loss['total_loss'].item()
            running_entropy += loss["cond_entropy"].item()

            pred_labels.extend(out_infer["y"].detach().cpu().numpy())
            true_labels.extend(labels.detach().cpu().numpy())

        train_loss = running_loss / len(dataloader.dataset)
        cond_entropy = running_entropy / len(dataloader.dataset)
        self.history["train_loss"].append(train_loss)
        self.history["train_accuracy"].append(self.get_accuracy(true_labels, pred_labels))
        self.history["train_cond_entropy"].append(-cond_entropy)
        self.current_epoch += 1
        return train_loss, out_infer

    def _evaluate(self):
        """
        Evaluate the model on the test dataset.

        Returns:
            tuple: A tuple containing the test loss and the inference results.
        """
        model = self.model.eval()
        criterion = self.criterion
        dataloader = self.test_loader
        device = self.device

        running_loss = 0.0
        running_entropy = 0.0
        pred_labels = []
        true_labels = []

        with torch.no_grad():
            for data, labels in dataloader:
                data = data.to(device)
                if self.transform_fn:
                    data = self.transform_fn(data)
                labels = labels.to(device)

                out_train, out_infer = model(data)
                loss = criterion(data, out_train)

                running_loss += loss['total_loss'].item()
                running_entropy += loss["cond_entropy"].item()

                pred_labels.extend(out_infer["y"].detach().cpu().numpy())
                true_labels.extend(labels.detach().cpu().numpy())

        test_loss = running_loss / len(dataloader.dataset)
        cond_entropy = running_entropy / len(dataloader.dataset)
        self.history["test_loss"].append(test_loss)
        self.history["test_accuracy"].append(self.get_accuracy(true_labels, pred_labels))
        self.history["test_cond_entropy"].append(-cond_entropy)
        return test_loss, out_infer

    def _get_n_ids_per_class(self, n):
        """
        Get n samples per class for evaluation.

        Args:
            n (int): Number of samples per class.

        Returns:
            numpy.ndarray: Array of random indices for each class.
        """

        if isinstance(self.test_loader.dataset,datasets.MNIST) or isinstance(self.test_loader.dataset,datasets.FashionMNIST):
            targets = self.test_loader.dataset.targets
        elif isinstance(self.test_loader.dataset,datasets.SVHN):
            targets = torch.Tensor(self.test_loader.dataset.labels)
        else: #CIFAR and TNFA
            targets = torch.Tensor(self.test_loader.dataset.targets)
        unique_values = targets.unique(return_counts=False)

        random_indices = []

        for value in unique_values:
            indices = torch.where(targets == value)[0]
            random_index = torch.randperm(len(indices))[:n]
            random_indices.extend(indices[random_index])

        # random_indices = torch.tensor(random_indices)
        random_indices = np.array(random_indices)
        return random_indices

    def _get_tracked_x_true(self):
        """
        Get the true images for the tracked IDs and store them in the history.
        """
        for true_id in self.tracked_ids:
            true_id = int(true_id)
            if isinstance(self.test_loader.dataset,datasets.MNIST):
                self.ids_history[true_id]["x_true"] = self.test_loader.dataset.data[true_id].cpu().numpy()
            if isinstance(self.test_loader.dataset,datasets.SVHN):
                self.ids_history[true_id]["x_true"] = self.test_loader.dataset.data[true_id]

    def _infer_tracked_ids(self):
        """
        Infer the latent variables for the tracked IDs and store them in the history.
        """
        model = self.model.eval()
        ids = self.tracked_ids
        dataset = self.test_loader.dataset
        device = self.device

        with torch.no_grad():
            data, labels = dataset.data[ids], dataset.targets[ids]
            data = data.to(device)
            if self.transform_fn:
                data = self.transform_fn(data)/255.0
            labels = labels.to(device)
            
            _, out_infer = model(data)

            for rel_id, true_id in enumerate(ids):
                true_id = int(true_id)
                for key in out_infer.keys():
                    temp_array = out_infer[key][rel_id].detach().cpu().numpy()
                    self.ids_history[true_id].setdefault(key, []).append(temp_array)

    def dump_to_json(self, data, file_path, indent=None):
        """
        Dump data to a JSON file.

        Args:
            data: Data to be saved as JSON.
            file_path (str): File path where the JSON data will be saved.
            indent (int, optional): Number of spaces for indentation. Defaults to None.
        """
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent, cls=NumpyEncoder)
        log.info(f"JSON data saved to: {file_path}")


class ProtoTrainer(Trainer):
    def __init__(self, model, optimizer, criterion, train_loader,
                 test_loader, device, path, track_ids=True, tracked_ids={},
                 n=1, binarize_x=False, transform_fn=flatten_mnist, scheduler=None, optimizer_class=None):
        super(ProtoTrainer, self).__init__(model, optimizer, criterion, train_loader,
                 test_loader, device, path, track_ids, tracked_ids, n, binarize_x, transform_fn)
        self.scheduler = scheduler
        self.optimizer_class = optimizer_class

    def get_accuracy(self, y_true, y_pred):
        """
        Get accuracy.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        corrected_pred = np.argmax(y_pred, axis=1)
        acc = np.mean(y_true == corrected_pred)
        return acc

    def _train_epoch(self):
        """
        Training loop
        """
        model = self.model.train()
        optimizer = self.optimizer
        criterion = self.criterion
        dataloader = self.train_loader
        device = self.device

        running_loss = 0.0
        running_entropy = 0.0
        running_cross_entropy = 0.0
        running_feat_cross_entropy = 0.0
        running_kl = 0.0
        running_recons = 0.0
        pred_labels = []
        true_labels = []

        for data, labels in tqdm(dataloader):
            data = data.to(device)

            if self.transform_fn:
                data = self.transform_fn(data)

            if self.binarize_x:
                batch = data.shape[0]
                thresholds = torch.rand((batch, 1)).to(data.device)
                data = torch.where(data > 1e-1, 1.0, 0.0)

            labels = labels.to(device)

            optimizer.zero_grad()
            out_train, out_infer = model(data)
            loss = criterion(data, labels, out_train, model.classif_layer[0].weight)
            loss['total_loss'].backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            if self.optimizer_class != None:
                optimizer_class.step()

            running_loss += loss['total_loss'].item()
            running_entropy += loss["cond_entropy"].item()
            running_cross_entropy += loss["classif_loss"].item()
            running_kl += loss["kl_loss"].item()
            running_recons += loss["recons_loss"].item()

            pred_labels.extend(out_train["pred_class"].detach().cpu().numpy())
            true_labels.extend(labels.detach().cpu().numpy())

        train_loss = running_loss / len(dataloader.dataset)
        cond_entropy = running_entropy / len(dataloader.dataset)
        cross_entropy = running_cross_entropy / len(dataloader.dataset)
        kl = running_kl / len(dataloader.dataset)
        recons = running_recons / len(dataloader.dataset)
        self.history["train_loss"].append(train_loss)
        self.history["train_accuracy"].append(self.get_accuracy(true_labels, pred_labels))
        self.history["train_cond_entropy"].append(-cond_entropy)
        self.history["train_cross_entropy"].append(cross_entropy)        
        self.history["train_recons"].append(recons)
        self.history["train_kl"].append(kl)
        self.current_epoch += 1
        if self.scheduler is not None:
            self.scheduler.step()
        return train_loss, out_infer

    def _evaluate(self):
        """
        Evaluate the model on the test dataset.

        Returns:
            tuple: A tuple containing the test loss and the inference results.
        """
        model = self.model.eval()
        criterion = self.criterion
        dataloader = self.test_loader
        device = self.device

        running_loss = 0.0
        running_entropy = 0.0
        running_cross_entropy = 0.0
        running_feat_cross_entropy = 0.0
        running_kl = 0.0
        running_recons = 0.0
        pred_labels = []
        true_labels = []

        with torch.no_grad():
            for data, labels in dataloader:
                data = data.to(device)
                if self.transform_fn:
                    data = self.transform_fn(data)
                labels = labels.to(device)

                out_train, out_infer = model(data)
                loss = criterion(data, labels, out_train, model.classif_layer[0].weight)

                running_loss += loss['total_loss'].item()
                running_entropy += loss["cond_entropy"].item()
                running_cross_entropy += loss["classif_loss"].item()
                running_kl += loss["kl_loss"].item()
                running_recons += loss["recons_loss"].item()

                pred_labels.extend(out_train["pred_class"].detach().cpu().numpy())
                true_labels.extend(labels.detach().cpu().numpy())

        test_loss = running_loss / len(dataloader.dataset)
        cond_entropy = running_entropy / len(dataloader.dataset)
        cross_entropy = running_cross_entropy / len(dataloader.dataset)
        kl = running_kl / len(dataloader.dataset)
        recons = running_recons / len(dataloader.dataset)
        self.history["test_loss"].append(test_loss)
        self.history["test_accuracy"].append(self.get_accuracy(true_labels, pred_labels))
        self.history["test_cond_entropy"].append(-cond_entropy)
        self.history["test_cross_entropy"].append(cross_entropy)
        self.history["test_recons"].append(recons)
        self.history["test_kl"].append(kl)
        return test_loss, out_infer

    def train(self, epochs):
        """
        Train the model for the specified number of epochs.

        Args:
            epochs (int): Number of epochs to train the model.
        """
        log.info(f"Training on {self.device}")
        if self.track_ids:
            if len(self.tracked_ids) == 0:
                self.tracked_ids = self._get_n_ids_per_class(self.n)
            self._get_tracked_x_true()

        for epoch in range(epochs):
            self._train_epoch()
            self._evaluate()

            # Track history for ids over epochs:
            if self.track_ids:
                self._infer_tracked_ids()

            train_loss = self.history["train_loss"][-1]
            test_loss = self.history["test_loss"][-1]
            train_acc = self.history["train_accuracy"][-1]
            test_acc = self.history["test_accuracy"][-1]
            train_ce = self.history["train_cross_entropy"][-1]
            test_ce = self.history["test_cross_entropy"][-1]
            train_kl = self.history["train_kl"][-1]
            test_kl = self.history["test_kl"][-1]
            train_recons = self.history["train_recons"][-1]
            test_recons = self.history["test_recons"][-1]
            train_cond_ent = self.history["train_cond_entropy"][-1]
            test_cond_ent = self.history["test_cond_entropy"][-1]

            log.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f},"
                     f"Test Loss: {test_loss:.4f}, Train CE:{train_ce:.4f}, Test CE:{test_ce:.4f}, Train Acc: {train_acc:.4f}, "
                     f"Test Acc: {test_acc:.4f}, Train KL:{train_kl:.4f}, Test KL:{test_kl:.4f}, Train Recons: {train_recons:.4f}, "
                     f"Test Recons: {test_recons:.4f}, Train cent:{train_cond_ent:.4f}, Test cent:{test_cond_ent:.4f}, ")

            if epoch>0 and epoch%100==0:
                checkpoint_path = self.path +str(epoch) + '/'
                os.makedirs(checkpoint_path, exist_ok=True)
                save_model(self.model, checkpoint_path, str(epoch))
                save_projections(self.model, self.test_loader, checkpoint_path+'test_')
                save_projections(self.model, self.train_loader, checkpoint_path+'train')
                save_images(self.model.visualize_all_components(), checkpoint_path,rgb=False)
                
        # After trainig actions:
        os.makedirs(self.path, exist_ok=True)
        history_path = os.path.join(self.path, "history.json")
        self.dump_to_json(self.history, history_path, indent=4)
        ids_history_path = os.path.join(self.path, "ids_history.json")
        self.dump_to_json(self.ids_history, ids_history_path)
        
        
    def _infer_tracked_ids(self):
        """
        Infer the latent variables for the tracked IDs and store them in the history.
        """
        model = self.model.eval()
        ids = self.tracked_ids
        dataset = self.test_loader.dataset
        device = self.device
        if isinstance(self.model.qy_x.h1,EncoderFC):
            encoder_type = 'FC'
        elif isinstance(self.model.qy_x.h1,EncoderCONV):
            encoder_type = 'CONV'
        elif isinstance(self.model.qy_x.h1,EncoderRGB):
            encoder_type = 'RGB'
        else:
            raise ValueError(f"Encoder type is not implemented.")
        
        with torch.no_grad():
            if isinstance(dataset,datasets.CIFAR10):
                data, labels = torch.Tensor(dataset.data[ids]).swapaxes(3,1), torch.Tensor(dataset.targets)[ids]
            if isinstance(dataset,datasets.SVHN):
                data, labels = torch.Tensor(dataset.data[ids]), torch.Tensor(dataset.labels[ids])
            if isinstance(dataset,datasets.MNIST): #works for FMNIST too
                data, labels = dataset.data[ids], dataset.targets[ids]
            if isinstance(dataset,datasets.ImageFolder): #TNFA100
                paths = np.array([t[0] for t in dataset.imgs])
                #data = np.array([dataset.loader(t) for t in paths[ids]])[:,:,:,0] #the loader converts to RGB by default
                data = np.array([dataset.loader(t) for t in paths[ids]],dtype=np.uint8)
                print(data.dtype)
                data, labels = torch.Tensor(data), torch.Tensor(dataset.targets[ids])

            data = data.to(device)
            if self.transform_fn:
                data = self.transform_fn(data)
            data = data/255.0
            labels = labels.to(device)

            if encoder_type=='CONV':
                _, out_infer = model(data.unsqueeze(1).float()) if isinstance(dataset,datasets.MNIST) else model(data.float())
            else:
                _, out_infer = model(data.float())

            for rel_id, true_id in enumerate(ids):
                true_id = int(true_id)
                for key in out_infer.keys():
                    temp_array = out_infer[key][rel_id].detach().cpu().numpy()
                    self.ids_history[true_id].setdefault(key, []).append(temp_array)
                    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    # Create a custom log message format
    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")

    # Get the existing logger (root logger in this case)
    logger = logging.getLogger()

    # Create a new handler and set the custom formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Remove existing handlers to avoid duplicate logging (optional)
    for old_handler in logger.handlers:
        logger.removeHandler(old_handler)

    # Add the new handler to the logger
    logger.addHandler(handler)

    k = 10
    encoder_type = "FC"
    input_size = 28*28
    hidden_size = 512
    latent_dim = 10     # 64

    model, criterion = get_model(k, encoder_type, input_size, hidden_size, latent_dim,
                                 recon_loss_type="BCE", eps=1e-8, model_name="GMVAE", loss_name="Loss",
                                 encoder_kwargs={"dropout": 0.1}, decoder_kwargs={})

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up data loaders
    # Define the transformation
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2048, shuffle=False, num_workers=8)

    # Move model to device
    model.to(device)
    # Define loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    trainer = Trainer(model, optimizer, criterion, train_loader, test_loader, device, path="./", transform_fn=flatten_mnist, binarize_x=True)
    trainer.train(2)
    print(trainer.ids_history.keys())
