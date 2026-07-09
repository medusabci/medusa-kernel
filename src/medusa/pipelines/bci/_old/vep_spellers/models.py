from abc import ABC, abstractmethod


class SpellerModel(ABC):

    def __init__(self):
        """Class constructor
        """
        super().__init__()
        # Settings
        self.settings = None
        self.configure()
        # Configuration
        self.is_configured = False
        self.is_built = False
        self.is_fit = False

    @abstractmethod
    def configure(self, **kwargs):
        """This function must be used to configure the model before calling
        build method. Class attribute settings attribute must be set with a dict
        """
        # Update state
        self.is_configured = True
        self.is_built = False
        self.is_fit = False

    @abstractmethod
    def build(self, *args, **kwargs):
        """This function builds the model
        """
        # Check errors
        if not self.is_configured:
            raise ValueError('Function configure must be called first!')
        # Update state
        self.is_built = True
        self.is_fit = False

    @abstractmethod
    def fit_online(self, times, signal, fs, channel_set, x_info, **kwargs):
        pass

    @abstractmethod
    def predict_online(self, times, signal, fs, channel_set, x_info, **kwargs):
        pass

    @abstractmethod
    def fit(self, dataset, **kwargs):
        pass

    @abstractmethod
    def predict(self, dataset, **kwargs):
        pass


