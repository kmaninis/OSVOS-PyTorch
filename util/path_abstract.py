class PathAbstract(object):

    @staticmethod
    def db_root_dir():
        raise NotImplementedError

    @staticmethod
    def save_root_dir():
        raise NotImplementedError

    @staticmethod
    def models_dir():
        raise NotImplementedError
