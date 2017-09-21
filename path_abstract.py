class PathAbstract(object):

    @staticmethod
    def db_root_dir():
        raise NotImplementedError

    @staticmethod
    def save_root_dir():
        raise NotImplementedError

    @staticmethod
    def exp_dir():
        raise NotImplementedError

    @staticmethod
    def is_custom_pytorch():
        raise NotImplementedError

    @staticmethod
    def custom_pytorch():
        raise NotImplementedError

    @staticmethod
    def is_custom_opencv():
        raise NotImplementedError

    @staticmethod
    def custom_opencv():
        raise NotImplementedError

    @staticmethod
    def models_dir():
        raise NotImplementedError
    #@staticmethod
    #def matlab_code():
    #    raise NotImplementedError
