from livenodes import Node
import numpy as np
import pickle


class LDASynthesis(Node.Node):
    """
    Predict quantized units based on LDA estimators
    """
    def __init__(self, params, select, name='LDASynthesis'):
        """
        Initializing all linear models for each spectral bin
        """
        super(LDASynthesis, self).__init__(name=name)
        self.estimators = pickle.loads(params)
        self.nb_bins = len(self.estimators)
        self.select = select

    def add_data(self, frame, data_id=0):
        """
        Reconstruct quantized bins based on an ECoG frame
        """
        frame = frame.reshape((1, -1))
        reco = np.empty(self.nb_bins)
        for spec_bin, est in enumerate(self.estimators):
            reco[spec_bin] = est.predict(frame[:, self.select])

        self.output_data(reco)
