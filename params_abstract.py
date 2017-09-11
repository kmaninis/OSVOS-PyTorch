class ParamsAbstract(object):

    @staticmethod
    def lr():
        return 1e-8

    @staticmethod
    def wd():
        return 0.0002

    @staticmethod
    def nAveGrad():
        return 5

    @staticmethod
    def nEpochs():
        return 2000