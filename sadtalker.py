import torch

from inference import main

class SadTalker:
    def __init__(self):
        self.driven_audio = "./examples/driven_audio/bus_chinese.wav"
        self.source_image = './examples/source_image/full_body_1.png'
        self.ref_eyeblink = None
        self.ref_pose = None
        self.checkpoint_dir = "./checkpoints"
        self.result_dir = "./results"
        self.result_name = None
        self.pose_style = 0
        self.batch_size = 2
        self.size = 256
        self.expression_scale = 1.
        self.input_yaw = None
        self.input_pitch = None
        self.input_roll = None
        self.enhancer = None
        self.background_enhancer = None
        self.cpu = False
        self.face3dvis = False
        self.still = False
        self.preprocess = "crop"
        self.verbose = False
        self.old_version = False

        self.net_recon = "resnet50"
        self.init_path = None
        self.use_last_fc = False
        self.bfm_folder = "./checkpoints/BFM_Fitting/"
        self.bfm_model = "BFN_model_front.mat"

        self.focal = 1015.
        self.center = 112.
        self.camera_d = 10.
        self.z_near = 5.
        self.z_far = 15.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run(self):
        return main(self)

if __name__ == "__main__":
    runner = SadTalker()
    runner.result_name = "my_name"
    runner.run()
