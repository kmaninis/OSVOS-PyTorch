from abc import ABCMeta, abstractmethod


class PathAbstract(object):
    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def db_root_dir():
        pass

    @staticmethod
    @abstractmethod
    def save_root_dir():
        pass

    @staticmethod
    @abstractmethod
    def exp_dir():
        pass

    @staticmethod
    @abstractmethod
    def is_custom_pytorch():
        pass

    @staticmethod
    @abstractmethod
    def custom_pytorch():
        pass

