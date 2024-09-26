import numpy as np
import threading
import queue

from os.path import join

class DataFetcher(threading.Thread):

    def __init__(self):
        super(DataFetcher, self).__init__()
        self.stopped = False
        self.queue = queue.Queue(64)
        self.index_list = list(range(20_000))
                    
        self.index = 0
        self.number = len(self.index_list)
        np.random.shuffle(self.index_list)
        self.pth = "/home/francois/Projects/data/training/Liver_patient_Paris_PCAresp"
            
    def load_data(self, idx):
        i = self.index_list[idx]
        paths = {
            "features": join(self.pth, "parameters", "mesh_pts.npy"),            # initial 3D coordinates
            "labels": join(self.pth, f'mesh_pts_{i}.npy'),                     # ground truth
            "img": join(self.pth, f'projection_{i}.npy'),                      # DRR input
            "img_label": join(self.pth, f'mesh_disp_2d_{i}.npy'),              # target deformation map
            "img_init": join(self.pth,"parameters", 'mesh_projection.npy'),      # Rest projected mesh
            "adj" : join(self.pth, "parameters", "mesh_adjacency.npy"),          # adj matrix size
            "rmax": join(self.pth, "parameters", "rmax.npy"),                    # rmax for projection    
            "camera_projection": join(self.pth, "parameters", "camera_projection.npy"), # projection matrix
            "faces": join(self.pth, "parameters", 'mesh_faces.npy')             # mesh faces
        }
        data = {k: np.load(v).squeeze().astype(np.float32) for k,v in paths.items()}

        ipos = np.einsum( 'ij, nj -> ni', data['camera_projection'][:3,:3], data['features'])
        ipos += data['camera_projection'][:3,3][None, :]
        ipos = ipos[:, :2] / ipos[:, 2:]
        
        data['labels'] = data['labels'].T

        center = np.mean(data['features'], 0)[None]

        n = len(data['features'])
        shapes = np.zeros((n, 3))
        for i in range(n):
            shapes[i] = data['features'][i] - center


        shift = 122.0
        constant = np.full(data['img'].shape, shift)
        img = np.stack((data['img_init'] * 255., constant, data['img'] * 255.), axis = 2)
        
        return data['adj'], data['features'], data['labels'], data['rmax'], data['camera_projection'], img, data['img_label'], shapes, ipos, data["faces"]

    def run(self):
        while self.index < 90000000 and not self.stopped:
            
            self.queue.put(self.load_data(self.index % self.number))
            self.index += 1

            if self.index % self.number == 0: 
                np.random.shuffle(self.index_list)

    def fetch(self):
        if self.stopped:
            return None
        return self.queue.get()

    def shutdown(self):
        self.stopped = True
        while not self.queue.empty():
            self.queue.get()
