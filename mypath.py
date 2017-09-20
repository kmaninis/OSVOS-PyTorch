from path_abstract import PathAbstract


class Path(PathAbstract):
    @staticmethod
    def db_root_dir():
        return '/media/eec/external/Databases/Segmentation/DAVIS/'

    @staticmethod
    def save_root_dir():
        return '/home/eec/Desktop/pytorch_experiments/osvos/deconv'

    @staticmethod
    def exp_dir():
        return '/home/eec/Desktop/pytorch_experiments/osvos/deconv'

    @staticmethod
    def is_custom_pytorch():
        return True

    @staticmethod
    def custom_pytorch():
        return "/home/eec/Documents/external/deep_learning/pytorch/build/lib.linux-x86_64-2.7"

    @staticmethod
    def is_custom_opencv():
        return False

    @staticmethod
    def custom_opencv():
        return None

