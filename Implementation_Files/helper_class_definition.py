from model_definitions import *
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import numpy as np
from torcheval.metrics import R2Score

class DL_Homework2_helper_class():
    """
    This class is a all in one Trainer class made to efficiently train all the different architectures described at one.
    It stors the dataset and has multiple dicts to concurently track the scores, weight updates and individual best models
    These dicts are saved in specific folders to be used in the final evaluation of the models

    The main functions implemented are:
        - validation_hyperparam_test: Uses the validation dataset to quickly test a matrix of hyper parameter combinations
        -- The output dictionary of this function Contains the validation R2 scores of every combination tested, and is used to find the
        -- hyperparameters that result in each architecture's best model

        - train_all_model_types: Trains all the models at once, used for both validation and the actual training steps
        -- Saves the best model by comparing R2 at each epoch
        -- The "Out_Dict" stores model information and epoch level scores to be saved and analyzed

        -nn_model_train_step: A helper function used to perform the backpropogation step as designed in pytorch
    """
    def __init__(self, x, y, device="cpu"):
        self.x = x
        self.y = y
        self.device= device
        self.out_dict = None
        self.optimizers = None
        self.validation_dict = None
        self.best = None

    def validation_hyperparam_test(self, loss_functions = [nn.MSELoss], optimizers = [torch.optim.SGD], lr_range = [0.0001, 0.001, 0.01, 0.1], epochs = 100, batch_size=4800):
        
        # main tracking output of the validation step
        self.validation_dict = {}
        self.out_dict = None

        for loss_function in loss_functions:
            for optimizer in optimizers:
                for lr in lr_range:

                    # Loop through all combinations of hyper parameters
                    self.train_all_model_types(loss_func=loss_function, optimizer=optimizer, lr=lr, epochs=epochs, batch_size=batch_size, validation=True)
                    self.validation_dict[loss_function._get_name()+"_"+optimizer.__name__+"_"+str(lr)] = self.out_dict["Scores"]

                    #reset state dict for next training interation
                    self.out_dict = None


    def train_all_model_types(self, save_path="",  loss_func=nn.CrossEntropyLoss, optimizer = torch.optim.SGD, lr=1e-1, epochs=10, batch_size = 373, validation = False):
        # Define out dictionary if it has not been defined before
        ## If it has been defined before then it will skip this step and continue training on the same models 
        if self.out_dict == None:
            self.out_dict = {}

            # Set up scores dict to track and save Loss/R2 for future analysis
            self.out_dict["Scores"] = {}

            # Here we set up all the models defined in the "model_definitions" module for the different architectures to be tested
            self.out_dict["DNNNet"] = DNNNet().to(self.device)
            self.out_dict["ConvNet"] = ConvNet().to(self.device) 
            
            self.best = {}

            for key in self.out_dict.keys(): 
                # Here we define what is needed to  be track in the "Scores" dict
                ## If validation, then we only need the R2 Validation score
                ## Else we need R2 and Loss scores on both the Training and Testing models
                ### These are used for forming the epoch graphs and tables for model analysis
                if validation == False:
                    if key != "Scores":
                        self.out_dict["Scores"][key] = {"train":[], "test":[], "loss":{"train":[], "test":[]}}
                        self.best[key] = [0, -1000]
                else:
                    if key != "Scores":
                        self.out_dict["Scores"][key] = {"valid":[]}
            
            #Pytorch requires separatly defined optimizers based on the model parameters
            # Here we set up the model specific optimizers for the training loop
            self.optimizers = {
                            key : optimizer(self.out_dict[key].parameters(), lr=lr) 
                                for key in self.out_dict.keys() 
                                if key!="Scores"
                        }
        
        #TQDM is used to show the progress of the model training by tracking the number of epochs
        for epoch in tqdm(range(epochs), maxinterval=epochs):

            # We take a sub batch of the total train dataset to speed up training iterations
            ## With a high enough number of epochs, using this method of batching seems to help model generatlisation when compared to using all of it 
            for i in range(int(self.x["train"].shape[0]/batch_size)-1):
                ind = np.array([k for k in range(self.x["train"].shape[0])])
                np.random.shuffle(ind)
                if (i+1)*batch_size <= self.x["train"].shape[0]:
                    x_batch = torch.Tensor(self.x["train"][i*batch_size : (i+1)*batch_size])
                    y_batch = torch.Tensor(self.y["train"][i*batch_size : (i+1)*batch_size])
                else:
                    x_batch = torch.Tensor(self.x["train"][i*batch_size : -1])
                    y_batch = torch.Tensor(self.y["train"][i*batch_size : -1])
                

                for key in self.out_dict.keys(): 
                    if key != "Scores":
                        if key == "DNNNet":
                            x_batch = x_batch.reshape([x_batch.shape[0], 784])
                        self.out_dict[key].train(True)
                        #Sequentially train each model architecture based on the current feature sampe and target y value
                        self.nn_model_train_step(self.out_dict[key], key, self.optimizers[key], x_batch, y_batch, loss_func=loss_func, lr=lr)
                        if key == "DNNNet":
                            x_batch = x_batch.reshape([x_batch.shape[0], 28 ,28])
            
            for key in self.out_dict.keys(): 
                if key != "Scores":
                    self.out_dict[key].eval()
                    with torch.no_grad():
                    # After each training Epoch, we update the Scores dict with what is required as defined before
                        if validation == False:
                            
                            #Current epoch training R2
                            train = self.out_dict[key](self.x["train"])
                            train = train.reshape([train.shape[0]])
                            acc = loss_func(train, self.y["train"])
                            self.out_dict["Scores"][key]["train"].append(acc)

                            #Current epoch test set R2
                            test = self.out_dict[key](torch.Tensor(self.x["test"]))
                            test = test.reshape([test.shape[0]])
                            metric = R2Score()
                            acc = loss_func(test, torch.Tensor(self.y["test"]))
                            self.out_dict["Scores"][key]["test"].append(acc)

                            if self.best[key][1] < self.out_dict["Scores"][key]["test"][-1]:
                                self.best[key][1] = self.out_dict["Scores"][key]["test"][-1]
                                self.best[key][0] = epoch
                                torch.save(self.out_dict[key], save_path+"//"+key+".pickle")
                        
                        else:
                            
                            #Current epoch validation set R2
                            train = self.out_dict[key](torch.Tensor(self.x["valid"]))
                            train = train.reshape([train.shape[0]])
                            acc = loss_func(train, self.y["valid"])
                            self.out_dict["Scores"][key]["valid"].append(acc)

    def nn_model_train_step(self, model, key, optimizer, x, y, loss_func=nn.CrossEntropyLoss(), lr=1e-1):
        optimizer.zero_grad()

        #Training loop step as defined for Pytorch models
        pred = model(x)
        output = loss_func(pred[0], y)
        output.backward()
        optimizer.step()
        

