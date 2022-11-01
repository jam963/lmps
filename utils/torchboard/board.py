from torch.utils.tensorboard import SummaryWriter

class Board(SummaryWriter):
    """
    Provides a more convenient way to log function outputs to TensorBoard when using PyTorch. Subclasses 
    SummaryWriter.
    """ 
    def __init__(self, log_dir='runs', writer_method="add_scalar"):
        super().__init__()
        self.steps = [0, 0, 0] # training, validation, testing steps
        self.state_names = ['Training', 'Validation', 'Testing']
        self.state = 0 # state index
        self.writer_method = getattr(self, writer_method) # SummaryWriter method to call, default is add_scalar
        
    def __call__(self, label, writer_method=None):
        steps = self.steps
        state_names = self.state_names
        writer_method = getattr(self, writer_method) if writer_method else self.writer_method
        get_state = self.get_state
        get_state_ix = self._get_state_ix
        def wrapper(func):
            def logger(*args, **kwargs):
                value = func(*args, **kwargs)
                state = get_state()
                state_ix = get_state_ix()
                writer_method(f'{label}/{state}', value, steps[state_ix])
                return value
            return logger    
        return wrapper

    def train(self):
        """ 
        Change Board state to "Training".
        """   
        self.state = 0

    def val(self):
        """ 
        Change Board state to "Validation".
        """  
        self.state = 1
    
    def test(self):
        """ 
        Change Board state to "Testing".
        """  
        self.state = 2

    def step(self):
        """
        Increments step count associated with current Board state.
        """
        self.steps[self.state] += 1

    def get_state(self):
        """
        Returns name of the current state for this Board.
        """
        return self.state_names[self.state]

    def _get_state_ix(self):
        return self.state