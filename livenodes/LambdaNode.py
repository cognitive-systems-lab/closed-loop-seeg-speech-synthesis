from . import Node


class LambdaNode(Node.Node):
    """
    Takes a stream of frames, applies a function to each, and passes the
    result back out.
    """
    def __init__(self, feature_function, name='LambdaNode'):
        """ Stores what function to apply. """
        super(LambdaNode, self).__init__(name=name)
        self.feature_function = feature_function
        
    def add_data(self, data_frame, data_id=0):
        """
        Callback for ready frames. Applies featureFunction to the
        input frames and passes them to all receivers.
        """
        self.output_data(self.feature_function(data_frame))
