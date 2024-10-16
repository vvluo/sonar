from collections import OrderedDict
import sys
from typing import Any, Dict, List
from torch import Tensor
import torch.nn as nn
from numpy import random
from utils.communication.comm_utils import CommunicationManager
from utils.log_utils import LogUtils
from algos.base_class import BaseClient, BaseServer
import os
import time


class CommProtocol:
    """
    Communication protocol tags for the server and clients
    """

    DONE = 0  # Used to signal the server that the client is done with local training
    START = 1  # Used to signal by the server to start the current round
    UPDATES = 2  # Used to send the updates from the server to the clients
    #FINISH = 3  # Used to signal the server to finish the current round for one client
    REQUEST = 4 # Used to request model parameters from the client
    INIT = 5
    LEND = 6 # Used to send updates, specifically the model weights learned by previous client

class ViewLayer(nn.Module):
    def __init__(self, shape):
        super(ViewLayer, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class SplitInferenceClient(BaseClient):
    def __init__(self, config: Dict[str, Any], comm_utils: CommunicationManager) -> None:
        super().__init__(config, comm_utils)
        self.config = config
        assert self.config is not None, "Config should be set when initializing"

        # self.dataloader = self.load_data(self.config) # see below
        # self.algo = self.load_algo(self.config, self.model_utils,self.dloader.train)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.view_layer = ViewLayer((-1, 512))
        self.first_model = nn.Sequential(*nn.ModuleList(list(self.model.children())[:self.config["split_layer"]]))

        try:
            config['log_path'] = f"{config['log_path']}/node_{self.node_id}"
            os.makedirs(config['log_path'])
        except FileExistsError:
            color_code = "\033[91m" # Red color
            reset_code = "\033[0m"   # Reset to default color
            print(f"{color_code}Log directory for the node {self.node_id} already exists in {config['log_path']}")
            print(f"Exiting to prevent accidental overwrite{reset_code}")
            sys.exit(1)

        config['load_existing'] = False
        self.client_log_utils = LogUtils(config)


    def local_train(self, round: int, **kwargs: Any):
        """
        Train the model locally
        """
        start_time = time.time()
        avg_loss, avg_accuracy = self.model_utils.train(
            self.model, self.optim, self.dloader, self.loss_fn, self.device
        )
        end_time = time.time()
        time_taken = end_time - start_time

        self.client_log_utils.log_console(
            "Client {} finished training with loss {:.4f}, accuracy {:.4f}, time taken {:.2f} seconds".format(self.node_id, avg_loss, avg_accuracy, time_taken)
            )
        self.client_log_utils.log_summary("Client {} finished training with loss {:.4f}, accuracy {:.4f}, time taken {:.2f} seconds".format(self.node_id, avg_loss, avg_accuracy, time_taken))

        self.client_log_utils.log_tb(f"train_loss/client{self.node_id}", avg_loss, round)
        self.client_log_utils.log_tb(f"train_accuracy/client{self.node_id}", avg_accuracy, round)
    

    def run_protocol(self):
        start_epochs = self.config.get("start_epochs", 0)
        total_epochs = self.config["epochs"]
        
        # print("client side model", self.first_model)

        for round in range(start_epochs, total_epochs):
            # print("client round ", round)
            repr = self.comm_utils.receive(self.server_node,tag=CommProtocol.LEND) # previous client
            # print("progress 1 round", round)
            if repr:
                self.model.load_state_dict(repr)
            # print("progress 2 round", round)

            
            
            def iterate_dataloader():
                for (data, target) in self.dloader:
                    yield (data, target)
            
            dataloader_iter = iterate_dataloader()

            batch_idx = 0
            while True:#for batch_idx in range(self.config["batch_size"]):
                # print("progress 3 round", round, " batch_idx ", batch_idx)
                batch_idx += 1
                try:
                    data, target = next(dataloader_iter)
                except StopIteration:
                    self.comm_utils.send(self.server_node,[[], []], CommProtocol.DONE)
                    break
                data = data.to(self.device)
                target = target.to(self.device)
                #print(data, target)
                intermediate = self.model_utils.forward_si_client(
                    self.first_model, self.global_pool, self.view_layer, self.optim, data, self.device)
                #print("here lies batch_idx ", batch_idx)
                #print(intermediate, target)# {"activations": intermediate, "labels", target} )
                self.comm_utils.send(self.server_node,[intermediate, target], CommProtocol.DONE)
                gradients = self.comm_utils.receive(self.server_node, CommProtocol.UPDATES)
                self.model_utils.backward_si_client(self.optim, intermediate, gradients, self.device)
            
            self.comm_utils.send(self.server_node, self.model.state_dict(), tag=CommProtocol.REQUEST)
            
            self.client_log_utils.log_summary("Client {} sending done signal to {}".format(self.node_id, self.server_node))
            
            self.client_log_utils.log_summary("Client {} waiting to get new model from {}".format(self.node_id, self.server_node))

            
            self.client_log_utils.log_summary("Client {} received new model from {}".format(self.node_id, self.server_node))
            

            # next round server requests the representation
            


class SplitInferenceServer(BaseServer):
    def __init__(self, config: Dict[str, Any], comm_utils: CommunicationManager) -> None:
        super().__init__(config, comm_utils)
        # self.set_parameters()
        self.config = config
        self.set_model_parameters(config)

        
        self.second_model = nn.Sequential(*nn.ModuleList(list(self.model.children())[self.config["split_layer"]:]))

        # self.init_model()
        self.model_save_path = "{}/saved_models/node_{}.pt".format(
            self.config["results_path"], self.node_id
        )
        self.folder_deletion_signal = config["folder_deletion_signal_path"]
    
    
    def get_representation(self, **kwargs: Any) -> OrderedDict[str, Tensor]:
        """
        Share the model weights
        """
        return self.model.state_dict() # type: ignore

    def set_representation(self, representation: OrderedDict[str, Tensor]):
        """
        Set the model weights
        """
        self.model.load_state_dict(representation)

    def client_sequencing(self, **kwargs: Any) -> List[int]:
        """
        Specifies order of training in terms of node_id, generated at random
        """
        client_indices = list(range(1, self.num_users + 1))
        random.shuffle(client_indices)
        return client_indices

    def single_round(self, round : int):#client_sequence : List[int],
        """
        Runs the whole training procedure
        """
        

        for client_index in range(1,self.num_users):#client_sequence:
            # print("client number ", client_index)
            # if round == 1:
            #     self.comm_utils.send(client_sequence[0],[],tag=CommProtocol.INIT) #empty communicaton on INIT
            #     # self.local_init()
            # else:
                #from previous round
            prev_client = client_index - 1#client_sequence[(client_index - 1) % len(client_sequence)]
            self.current_client = client_index#client_sequence[client_index]
            self.log_utils.log_console("Training with client {}".format(self.current_client))
            self.log_utils.log_summary("Training with client {}".format(self.current_client))
            if client_index == 1:
                repr = []
                if round != 0:
                    # print("alternate blows")
                    repr = self.comm_utils.receive(self.num_users-1, tag=CommProtocol.REQUEST)
                self.comm_utils.send(self.current_client, repr, tag=CommProtocol.LEND)
            else:
                # print("gathering for client ", client_index)
                self.log_utils.log_console("Gathering representation for client {}".format(self.current_client))
                base_network = self.comm_utils.receive(prev_client, tag=CommProtocol.REQUEST)
                # print("I'll tell you a story")
                self.comm_utils.send(self.current_client, base_network, tag=CommProtocol.LEND) #Should this CommProtocol differ from the one with gradient updates
                self.log_utils.log_console("Sent representation for client {}".format(self.current_client))
                # print("about a little bird")
            while True: #for batch_idx in range(self.config['batch_size']):
                intermediate, target = self.comm_utils.receive(self.current_client, tag=CommProtocol.DONE)
                if intermediate == []:
                    break
                # intermediate = self.global_pool(intermediate)
                # intermediate = intermediate.view(-1,intermediate.size(1))
                output = self.model_utils.forward_si_server(self.second_model, self.optim, intermediate, self.device)
                gradients = self.model_utils.backward_si_server(self.optim, intermediate, output, target, self.loss_fn, self.device)
                self.comm_utils.send(self.current_client, gradients, tag=CommProtocol.UPDATES) #sends gradients to clients
            

        #Remove the signal file after confirming that all client paths have been created
        if os.path.exists(self.folder_deletion_signal):
            os.remove(self.folder_deletion_signal)

    def test(self, **kwargs: Any) -> List[float]:
        """
        Test the model on the server
        """
        start_time = time.time()
        test_loss, test_acc = self.model_utils.test(
            self.model, self._test_loader, self.loss_fn, self.device
        )
        end_time = time.time()
        time_taken = end_time - start_time
        # TODO save the model if the accuracy is better than the best accuracy
        # so far
        if test_acc > self.best_acc:
            self.best_acc = test_acc
            self.model_utils.save_model(self.model, self.model_save_path)
        return [test_loss, test_acc, time_taken]


    def run_protocol(self):
        # print("server side model", self.second_model)
        # print(self.config["split_layer"])

        client_sequence = self.client_sequencing()
        self.current_client = client_sequence[0]

        train_dset = self.dset_obj.train_dset
        print("here is a training dataset for the server: ", train_dset)
        batch_size = self.config["training_batch_size"]
        # self._train_loader = DataLoader(train_dset, batch_size=batch_size)

        self.log_utils.log_console("Starting splitNN")
        start_epochs = self.config.get("start_epochs", 0)
        total_epochs = self.config["epochs"]
        # for epoch in range(start_epochs, total_epochs):
        #     status = self.comm_utils.receive(0, tag=self.tag.START)
        # self.comm_utils.send(client_sequence[0],[],tag=CommProtocol.INIT) # unnecessary since each client auto_inits (may be inefficient)
        for round in range(start_epochs, total_epochs):
            self.log_utils.log_console("Starting round {}".format(round))
            self.log_utils.log_summary("Starting round {}".format(round))
            self.single_round(round) #client_sequence
            self.log_utils.log_console("Server testing the model")

            self.log_utils.log_console("Testing Voided")
            # loss, acc, time_taken = self.test()
            # self.log_utils.log_tb(f"test_acc/clients", acc, round)
            # self.log_utils.log_tb(f"test_loss/clients", loss, round)
            # self.log_utils.log_console("Round: {} test_acc:{:.4f}, test_loss:{:.4f}, time taken {:.2f} seconds".format(round, acc, loss, time_taken))
            
            
            # self.log_utils.log_summary("Round: {} test_acc:{:.4f}, test_loss:{:.4f}, time taken {:.2f} seconds".format(round, acc, loss, time_taken))
            self.log_utils.log_console("Round {} complete".format(round))
            self.log_utils.log_summary("Round {} complete".format(round,))
