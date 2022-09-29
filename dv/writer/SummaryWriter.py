class SummaryWriter(object):
    """Writes entries directly to event files in the log_dir to be
    consumed by TensorBoard.

    The `SummaryWriter` class provides a high-level API to create an event file
    in a given directory and add summaries and events to it. The class updates the
    file contents asynchronously. This allows a training program to call methods
    to add data to the file directly from the training loop, without slowing down
    training.
    """

    def __init__(self, log_dir):
        """Creates a `SummaryWriter` that will write out events and summaries
        to the event file.

        Args:
            log_dir (string): Save directory location.
        """
        log_dir = str(log_dir)
        self.log_dir = log_dir

    def get_logdir(self):
        """Returns the directory where event file will be written."""
        return self.log_dir
    
    def add_training_data(self, dataset, transform):
        pass

    def add_testing_data(self, dataset, transform):
        pass

    def add_epoch_data(self, epoch, state_dict, idxs):
        pass

    def add_config(self,):
        pass

    def add_subject_model(self,):
        pass

    def add_iteration_structure(self,):
        pass