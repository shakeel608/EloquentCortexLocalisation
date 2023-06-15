#Author: Dr. Shakeel Ahmad Sheikh
#Affiliation: PostDoctoral Research Scientist, CITEC, University of Bielefeld, Germany
#Date: June 1, 2023
#Description: This script describes trainer wrapper for Eloquent Cortex Tumor Detection


from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
import torch
from monai.metrics import DiceMetric
import wandb
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
)


class TrainWrapper():
    def __init__(self, train_loader, val_loader, model, optimizer, loss_function, device, train_ds, num_epochs):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_interval = 1
        self.best_metric = -1
        self.best_metric_epoch = -1
        self.epoch_loss_values = list()
        self.metric_values = list()
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_ds = train_ds
        self.num_epochs = num_epochs
        self.loss_function = loss_function
        self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        #self.dice_metric_train = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    def train(self, epoch):
        self.model.train()
        epoch_loss = 0
        step = 0
        for batch_data in self.train_loader:
            step += 1
            inputs, labels = batch_data["img"].to(self.device), batch_data["seg"].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(self.train_ds) // self.train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            #self.dice_metric_train(y_pred=outputs, y=labels)
            #print("Tran Dice Metric",metric_train);break
        epoch_loss /= step
        #metric_train = self.dice_metric_train.aggregate().item()
        #print("Train Dice Score =={}".format(metric_train))
        self.epoch_loss_values.append(epoch_loss)
        #wandb.log({"Train Loss": epoch_loss}, step=epoch+1)
        #wandb.log({"Train Dice Score": metric_train}, step=epoch+1)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        
    def val(self, epoch):
        self.model.eval()
        val_epoch_loss = 0
        step = 0
        if (epoch + 1) % self.val_interval == 0:
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                #print("self dice", self.dice_metric.aggregate().item())#;exit(0)
                for val_data in self.val_loader:
                    step +=1
                    val_images, val_labels = val_data["img"].to(self.device), val_data["seg"].to(self.device)
                    #loss = self.loss_function(val_images, val_labels)
                    #val_epoch_loss +=loss.item()
                    
                    roi_size = (96, 96, 96)
                    sw_batch_size = 10
                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, self.model)
                    #print("val outut shape", val_outputs.shape)
                    val_outputs = [self.post_trans(i) for i in decollate_batch(val_outputs)]
                    #print("Batch ={} and val outut shape={}".format(len(val_outputs), val_outputs[0].shape)); exit(0)
                    # compute metric for current iteration
                    self.dice_metric(y_pred=val_outputs, y=val_labels)
                # aggregate the final mean dice result
                metric = self.dice_metric.aggregate().item()
                print("Val Dice Score =={}".format(metric))
                # reset the status for next validation round
                #wandb.log({" Val Dice Score": metric}, step=epoch+1)
                self.dice_metric.reset()

                #Val Dice Loss
                #val_epoch_loss /=step
                #wandb.log({"Val Dice Loss ": val_epoch_loss}, step=epoch+1)
                #print("Val Epoch Loss", val_epoch_loss)
                self.metric_values.append(metric)
                if metric > self.best_metric:
                    self.best_metric = metric
                    self.best_metric_epoch = epoch + 1
                    torch.save(self.model.state_dict(), "best_metric_model_segmentation3d_dict.pth")
                    print("saved new best metric model")
                print("On Val. Set, current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(epoch + 1, metric, self.best_metric, self.best_metric_epoch))        
        return self.best_metric, self.best_metric_epoch
