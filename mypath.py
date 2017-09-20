from path_abstract import PathAbstract


class Path(PathAbstract):
    @staticmethod
    def db_root_dir():
        return '/scratch_net/zoidberg_second/csergi/Databases/DAVIS'

    @staticmethod
    def save_root_dir():
        return '/srv/glusterfs/kmaninis/pytorch_experiments/osvos/deconv'

    @staticmethod
    def exp_dir():
        return '/srv/glusterfs/kmaninis/pytorch_experiments/osvos/deconv'

    @staticmethod
    def is_custom_pytorch():
        return True

    @staticmethod
    def custom_pytorch():
        return "/home/kmaninis/scratch_net/reinhold/Kevis/Software/apps/pytorch/build/lib.linux-x86_64-2.7"

    @staticmethod
    def is_custom_opencv():
        return False

    @staticmethod
    def custom_opencv():
        return None

