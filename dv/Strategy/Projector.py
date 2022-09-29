"""The Projector class for visualization, serve as a helper module for evaluator and visualizer"""
from abc import ABC, abstractmethod

import os
import numpy as np
import torch

class ProjectorAbstractClass(ABC):

    @abstractmethod
    def __init__(self, vis_model, content_path, *args, **kwargs):
        pass

    @abstractmethod
    def load(self, *args, **kwargs):
        pass

    @abstractmethod
    def batch_project(self, *args, **kwargs):
        pass

    @abstractmethod
    def individual_project(self, *args, **kwargs):
        pass

    @abstractmethod
    def batch_inverse(self, *args, **kwargs):
        pass

    @abstractmethod
    def individual_inverse(self, *args, **kwargs):
        pass


class Projector(ProjectorAbstractClass):
    def __init__(self, vis_model, content_path, segments, device) -> None:
        self.content_path = content_path
        self.vis_model = vis_model
        self.segments = segments    #[(1,6),(6, 15),(15,42),(42,200)]
        self.DEVICE = device
        self.current_range = (-1,-1)

    def load(self, iteration):
        # [s,e)
        init_e = self.segments[-1][1]
        if (iteration >= self.current_range[0] and iteration <self.current_range[1]) or (iteration == init_e and self.current_range[1] == init_e):
            print("Same range as current visualization model...")
            return 
        # else
        for i in range(len(self.segments)):
            s = self.segments[i][0]
            e = self.segments[i][1]
            # range [s,e)
            if (iteration >= s and iteration < e) or (iteration == init_e and e == init_e):
                idx = i
                break
        file_path = os.path.join(self.content_path, "Model", "tnn_hybrid_{}.pth".format(idx))
        save_model = torch.load(file_path, map_location="cpu")
        self.vis_model.load_state_dict(save_model["state_dict"])
        self.vis_model.to(self.DEVICE)
        self.vis_model.eval()
        self.current_range = (s, e)
        print("Successfully load the visualization model for range ({},{})...".format(s,e))


    def batch_project(self, iteration, data):
        self.load(iteration)
        embedding = self.vis_model.encoder(torch.from_numpy(data).to(dtype=torch.float32, device=self.DEVICE)).cpu().detach().numpy()
        return embedding
    
    def individual_project(self, iteration, data):
        self.load(iteration)
        embedding = self.vis_model.encoder(torch.from_numpy(np.expand_dims(data, axis=0)).to(dtype=torch.float32, device=self.DEVICE)).cpu().detach().numpy()
        return embedding.squeeze(axis=0)
    
    def batch_inverse(self, iteration, embedding):
        self.load(iteration)
        data = self.vis_model.decoder(torch.from_numpy(embedding).to(dtype=torch.float32, device=self.DEVICE)).cpu().detach().numpy()
        return data
    
    def individual_inverse(self, iteration, embedding):
        self.load(iteration)
        data = self.vis_model.decoder(torch.from_numpy(np.expand_dims(embedding, axis=0)).to(dtype=torch.float32, device="cpu")).cpu().detach().numpy()
        return data.squeeze(axis=0)