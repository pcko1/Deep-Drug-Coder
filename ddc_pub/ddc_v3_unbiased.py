import os
os.environ[
    "TF_CPP_MIN_LOG_LEVEL"
] = "3"  # Suppress UserWarning of TensorFlow while loading the model

import numpy as np
from datetime import datetime
from functools import wraps
import shutil, zipfile, tempfile, pickle

from tensorflow.keras.layers import (
    Input,
    Concatenate,
    Dense,
    TimeDistributed,
    BatchNormalization,
)
from tensorflow.compat.v1.keras.layers import (
    CuDNNLSTM as LSTM,
) 
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.utils import multi_gpu_model, plot_model

# Custom dependencies
from molvecgen import SmilesVectorizer

from ddc_pub.generators import SmilesGenerator2
from ddc_pub.custom_callbacks import ModelAndHistoryCheckpoint, LearningRateSchedule


def timed(func):
    """
    Timer decorator to benchmark functions.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        tstart = datetime.now()
        result = func(*args, **kwargs)
        elapsed = (datetime.now() - tstart).microseconds / 1e6
        print("Elapsed time: %.3f seconds." % elapsed)
        return result

    return wrapper


class DDC:
    def __init__(self, **kwargs):
        """
        # Arguments
            kwargs:
                x            : model input                                                  - np.ndarray of np.bytes_ or np.float64
                y            : model output                                                 - np.ndarray of np.bytes_
                model_name   : model filename to load                                       - string
                dataset_info : dataset information including name, maxlen and charset       - hdf5
                noise_std    : standard deviation of the noise layer in the latent space    - float
                lstm_dim     : size of LSTM RNN layers                                      - int
                dec_layers   : number of decoder layers                                     - int
                td_dense_dim : size of TD Dense layers inbetween the LSTM ones
                               to suppress network size                                     - int
                batch_size   : the network's batch size                                     - int
                codelayer_dim: dimensionality of the latent space or number of descriptors  - int
                
                
        # Examples of __init__ usage
            To *train* a blank model with encoder (autoencoder):
                model = ddc.DDC(x              = mols,
                                y              = mols,
                                dataset_info   = info,
                                noise_std      = 0.1,
                                lstm_dim       = 256,
                                dec_layers     = 3,
                                td_dense_dim   = 0,
                                batch_size     = 128,
                                codelayer_dim  = 128)
            
            To *train* a blank model without encoder:
                model = ddc.DDC(x              = descriptors,
                                y              = mols,
                                dataset_info   = info,
                                noise_std      = 0.1,
                                lstm_dim       = 256,
                                dec_layers     = 3,
                                td_dense_dim   = 0,
                                batch_size     = 128)
                                
            To *re-train* a saved model with encoder (autoencoder):
                model = ddc.DDC(x              = mols,
                                y              = mols,
                                model_name     = saved_model_name)
            
            To *re-train* a saved model without encoder:
                model = ddc.DDC(x              = descriptors,
                                y              = mols,
                                model_name     = saved_model_name)
                
            To *test* a saved model:
                model = ddc.DDC(model_name     = saved_model_name)

        """

        # Identify the mode to start the model in
        if "x" in kwargs:
            x = kwargs.get("x")
            if "model_name" not in kwargs:
                self.__mode = "train"
            else:
                self.__mode = "retrain"
        elif "model_name" in kwargs:
            self.__mode = "test"
        else:
            raise NameError("Cannot infer mode from arguments.")

        print("Initializing model in %s mode." % self.__mode)

        if self.mode == "train":
            # Infer input type from type(x)
            if type(x[0]) == np.bytes_:
                print("Input type is 'binary mols'.")
                self.__input_type = "mols"  # binary RDKit mols
            else:
                print("Check input type.")
                self.__input_type = "other"  # other molecular descriptors

            self.__maxlen = (
                kwargs.get("dataset_info")["maxlen"] + 10
            )  # Extend maxlen to avoid breaks in training
            self.__charset = kwargs.get("dataset_info")["charset"]
            self.__dataset_name = kwargs.get("dataset_info")["name"]
            self.__lstm_dim = kwargs.get("lstm_dim", 256)
            self.__h_activation = kwargs.get("h_activation", "relu")
            self.__bn = kwargs.get("bn", True)
            self.__bn_momentum = kwargs.get("bn_momentum", 0.9)
            self.__noise_std = kwargs.get("noise_std", 0.01)
            self.__td_dense_dim = kwargs.get(
                "td_dense_dim", 0
            )  # >0 squeezes RNN connections with Dense sandwiches
            self.__batch_size = kwargs.get("batch_size", 256)
            self.__dec_layers = kwargs.get("dec_layers", 2)

            self.__codelayer_dim = kwargs.get("codelayer_dim", 128)

            # Create the left/right-padding vectorizers
            self.__smilesvec1 = SmilesVectorizer(
                canonical=False,
                augment=True,
                maxlength=self.maxlen,
                charset=self.charset,
                binary=True,
            )

            self.__smilesvec2 = SmilesVectorizer(
                canonical=False,
                augment=True,
                maxlength=self.maxlen,
                charset=self.charset,
                binary=True,
                leftpad=False,
            )

            # self.train_gen.next() #This line is needed to set train_gen.dims (to be fixed in HetSmilesGenerator)
            self.__input_shape = self.smilesvec1.dims
            self.__dec_dims = list(self.smilesvec1.dims)
            self.__dec_dims[0] = self.dec_dims[0] - 1
            self.__dec_input_shape = self.dec_dims
            self.__output_len = self.smilesvec1.dims[0] - 1
            self.__output_dims = self.smilesvec1.dims[-1]

            # Build data generators
            self.__build_generators(x)

            # Build full model out of the sub-models
            self.__build_model()

        # Retrain or Test mode
        else:
            self.__model_name = kwargs.get("model_name")

            # Load the model
            self.__load(self.model_name)

            if self.mode == "retrain":
                # Build data generators
                self.__build_generators(x)

        

        # Show the resulting full model
        print(self.model.summary())

    """
    Architecture properties.
    """

    @property
    def lstm_dim(self):
        return self.__lstm_dim

    @property
    def h_activation(self):
        return self.__h_activation

    @property
    def bn(self):
        return self.__bn

    @property
    def bn_momentum(self):
        return self.__bn_momentum

    @property
    def noise_std(self):
        return self.__noise_std

    @property
    def td_dense_dim(self):
        return self.__td_dense_dim

    @property
    def batch_size(self):
        return self.__batch_size

    @property
    def dec_layers(self):
        return self.__dec_layers

    @property
    def codelayer_dim(self):
        return self.__codelayer_dim

    @property
    def steps_per_epoch(self):
        return self.__steps_per_epoch

    @property
    def validation_steps(self):
        return self.__validation_steps

    @property
    def input_shape(self):
        return self.__input_shape

    @property
    def dec_dims(self):
        return self.__dec_dims

    @property
    def dec_input_shape(self):
        return self.__dec_input_shape

    @property
    def output_len(self):
        return self.__output_len

    @property
    def output_dims(self):
        return self.__output_dims

    @property
    def batch_input_length(self):
        return self.__batch_input_length

    #@batch_input_length.setter
    #def batch_input_length(self, value):
    #    self.__batch_input_length = value
    #    self.__build_sample_model(batch_input_length=value)

    """
    Models.
    """

    @property
    def sample_model(self):
        return self.__sample_model

    @property
    def multi_sample_model(self):
        return self.__multi_sample_model

    @property
    def model(self):
        return self.__model

    """
    Train properties.
    """

    @property
    def epochs(self):
        return self.__epochs

    @property
    def clipvalue(self):
        return self.__clipvalue

    @property
    def lr(self):
        return self.__lr

    @property
    def h(self):
        return self.__h

    """
    Other properties.
    """

    @property
    def mode(self):
        return self.__mode

    @property
    def dataset_name(self):
        return self.__dataset_name

    @property
    def model_name(self):
        return self.__model_name

    @property
    def input_type(self):
        return self.__input_type

    @property
    def maxlen(self):
        return self.__maxlen

    @property
    def charset(self):
        return self.__charset

    @property
    def smilesvec1(self):
        return self.__smilesvec1

    @property
    def smilesvec2(self):
        return self.__smilesvec2

    @property
    def train_gen(self):
        return self.__train_gen

    @property
    def valid_gen(self):
        return self.__valid_gen

    """
    Private methods.
    """

    def __build_generators(self, x, split=0.81050343):
        """
            Build data generators to be used in (re)training.
            """

        # Split dataset into train and validation sets
        cut = int(split * len(x))
        x_train = x[:cut]
        x_valid = x[cut:]

        self.__train_gen = SmilesGenerator2(
            x_train,
            None,
            self.smilesvec1,
            self.smilesvec2,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.__valid_gen = SmilesGenerator2(
            x_valid,
            None,
            self.smilesvec1,
            self.smilesvec2,
            batch_size=self.batch_size,
            shuffle=True,
        )

        # Calculate number of batches per training/validation epoch
        train_samples = len(x_train)
        valid_samples = len(x_valid)
        self.__steps_per_epoch = train_samples // self.batch_size
        self.__validation_steps = valid_samples // self.batch_size

        print(
            "Model received %d train samples and %d validation samples."
            % (train_samples, valid_samples)
        )

    def __build_model(self):
        """
        RNN that generates random SMILES strings.
        """

        # This is the start character padded OHE smiles for teacher forcing
        decoder_inputs = Input(shape=self.dec_input_shape, name="Decoder_Inputs")

        # I/O tensor of the LSTM layers
        x = decoder_inputs

        for dec_layer in range(self.dec_layers):
            # RNN layer
            decoder_lstm = LSTM(
                self.lstm_dim,
                return_sequences=True,
                name="Decoder_LSTM_" + str(dec_layer),
            )

            x = decoder_lstm(x)

            if self.bn:
                x = BatchNormalization(
                    momentum=self.bn_momentum, name="BN_Decoder_" + str(dec_layer)
                )(x)

            # Squeeze LSTM interconnections using Dense layers
            if self.td_dense_dim > 0:
                x = TimeDistributed(
                    Dense(self.td_dense_dim), name="Time_Distributed_" + str(dec_layer)
                )(x)

        # Final Dense layer to return soft labels (probabilities)
        outputs = Dense(self.output_dims, activation="softmax", name="Dense_Decoder")(x)

        # Define the batch_model
        self.__model = Model(inputs=[decoder_inputs], outputs=[outputs])

        # Name it!
        self.__model._name = "model"

    def __build_sample_model(self, batch_input_length) -> dict:
        """
        Model that predicts a single OHE character.
        This model is generated from the modified config file of the self.batch_model.

        Returns:
            The dictionary of the configuration.
        """

        self.__batch_input_length = batch_input_length

        # Get the configuration of the batch_model
        config = self.model.get_config()

        # Keep only the "Decoder_Inputs" as single input to the sample_model
        config["input_layers"] = [config["input_layers"][0]]

        # Find decoder states that are used as inputs in batch_model and remove them
        idx_list = []
        for idx, layer in enumerate(config["layers"]):

            if "Decoder_State_" in layer["name"]:
                idx_list.append(idx)

        # Pop the layer from the layer list
        # Revert indices to avoid re-arranging after deleting elements
        for idx in sorted(idx_list, reverse=True):
            config["layers"].pop(idx)

        # Remove inbound_nodes dependencies of remaining layers on deleted ones
        for layer in config["layers"]:
            idx_list = []

            try:
                for idx, inbound_node in enumerate(layer["inbound_nodes"][0]):
                    if "Decoder_State_" in inbound_node[0]:
                        idx_list.append(idx)
            # Catch the exception for first layer (Decoder_Inputs) that has empty list of inbound_nodes[0]
            except:
                pass

            # Pop the inbound_nodes from the list
            # Revert indices to avoid re-arranging
            for idx in sorted(idx_list, reverse=True):
                layer["inbound_nodes"][0].pop(idx)

        # Change the batch_shape of input layer
        config["layers"][0]["config"]["batch_input_shape"] = (
            batch_input_length,
            1,
            self.dec_input_shape[-1],
        )

        # Finally, change the statefulness of the RNN layers
        for layer in config["layers"]:
            if "Decoder_LSTM_" in layer["name"]:
                layer["config"]["stateful"] = True
                # layer["config"]["return_sequences"] = True

        # Define the sample_model using the modified config file
        sample_model = Model.from_config(config)

        # Copy the trained weights from the trained batch_model to the untrained sample_model
        for layer in sample_model.layers:
            # Get weights from the batch_model
            weights = self.model.get_layer(layer.name).get_weights()
            # Set the weights to the sample_model
            sample_model.get_layer(layer.name).set_weights(weights)

        if batch_input_length == 1:
            self.__sample_model = sample_model

        elif batch_input_length > 1:
            self.__multi_sample_model = sample_model

        return config

    def __load(self, model_name):
        """
        Load complete model from a zip file.
        To be called within __init__.
        """

        print("Loading model.")
        tstart = datetime.now()

        # Temporary directory to extract the zipped information
        with tempfile.TemporaryDirectory() as dirpath:

            # Unzip the directory that contains the saved model(s)
            with zipfile.ZipFile(model_name + ".zip", "r") as zip_ref:
                zip_ref.extractall(dirpath)

            # Load metadata
            metadata = pickle.load(open(dirpath + "/metadata.pickle", "rb"))

            # Re-load metadata
            self.__dict__.update(metadata)

            # Load the model
            self.__model = load_model(dirpath + "/model.h5")

            # Build sample_model out of the trained batch_model
            self.__build_sample_model(batch_input_length=1)  # Single-output model
            self.__build_sample_model(
                batch_input_length=256
            )  # Multi-output model

        print("Loading finished in %i seconds." % ((datetime.now() - tstart).seconds))

    """
    Public methods.
    """

    def fit(
        self,
        model_name,
        epochs,
        lr,
        mini_epochs,
        patience,
        gpus=1,
        workers=1,
        use_multiprocessing=False,
        verbose=2,
        max_queue_size=10,
        clipvalue=0,
        save_period=5,
        checkpoint_dir="/",
        lr_decay=False,
        lr_warmup=False,
        sch_epoch_to_start=500,
        sch_last_epoch=999,
        sch_lr_init=1e-3,
        sch_lr_final=1e-6,
    ):
        """
        Fit the full model to the training data.
        Supports multi-gpu training if gpus set to >1.
        
        # Arguments
            kwargs:
                model_name         : base name for the checkpoints                                       - string
                epochs             : number of epochs to train in total                                  - int
                lr                 : initial learning rate of the training                               - float
                mini_epochs        : number of dividends of an epoch (==1 means no mini_epochs)          - int
                patience           : minimum consecutive mini_epochs of stagnated learning rate to consider 
                                     before lowering it                                                  - int
                gpus               : number of gpus to use for multi-gpu training (==1 means single gpu) - int
                workers            : number of CPU workers                                               - int
                use_multiprocessing: flag for Keras multiprocessing                                      - boolean
                verbose            : verbosity of the training                                           - int
                max_queue_size     : max size of the generator queue                                     - int
                clipvalue          : value of gradient clipping                                          - float
                save_period        : mini_epochs every which to checkpoint the model                     - int
                checkpoint_dir     : directory to store the checkpoints                                  - string
                lr_decay           : flag to use exponential decay of learning rate                      - boolean
                lr_warmup          : flag to use warmup for transfer learning                            - boolean
        """

        # Get parameter values if specified
        self.__epochs = epochs
        self.__lr = lr
        self.__clipvalue = clipvalue

        # Optimizer
        if clipvalue > 0:
            print("Using gradient clipping %.2f." % clipvalue)
            opt = Adam(lr=self.lr, clipvalue=self.clipvalue)

        else:
            opt = Adam(lr=self.lr)

        checkpoint_file = (
            checkpoint_dir + "%s--{epoch:02d}--{val_loss:.4f}--{lr:.7f}" % model_name
        )

        # If model is untrained, history is blank
        try:
            history = self.h

        # Else, append the history
        except:
            history = {}

        mhcp = ModelAndHistoryCheckpoint(
            filepath=checkpoint_file,
            model_dict=self.__dict__,
            monitor="val_loss",
            verbose=1,
            mode="min",
            period=save_period,
            history=history
        )

        # Training history
        self.__h = mhcp.history

        if lr_decay:
            lr_schedule = LearningRateSchedule(
                epoch_to_start=sch_epoch_to_start,
                last_epoch=sch_last_epoch,
                lr_init=sch_lr_init,
                lr_final=sch_lr_final,
            )

            lr_scheduler = LearningRateScheduler(
                schedule=lr_schedule.exp_decay, verbose=1
            )

            callbacks = [lr_scheduler, mhcp]

        elif lr_warmup:
            lr_schedule = LearningRateSchedule(
                epoch_to_start=sch_epoch_to_start,
                last_epoch=sch_last_epoch,
                lr_init=sch_lr_init,
                lr_final=sch_lr_final,
            )

            lr_scheduler = LearningRateScheduler(
                schedule=lr_schedule.warmup, verbose=1
            )

            callbacks = [lr_scheduler, mhcp]

        else:
            rlr = ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=patience,
                min_lr=1e-6,
                verbose=1,
                min_delta=1e-4,
            )

            callbacks = [rlr, mhcp]

        # Inspect training parameters at the start of the training
        self.summary()

        # Parallel training on multiple GPUs
        if gpus > 1:
            parallel_model = multi_gpu_model(self.model, gpus=gpus)
            parallel_model.compile(loss="categorical_crossentropy", optimizer=opt)
            # This `fit` call will be distributed on all GPUs.
            # Each GPU will process (batch_size/gpus) samples per batch.
            parallel_model.fit_generator(
                self.train_gen,
                steps_per_epoch=self.steps_per_epoch / mini_epochs,
                epochs=mini_epochs * self.epochs,
                validation_data=self.valid_gen,
                validation_steps=self.validation_steps / mini_epochs,
                callbacks=callbacks,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                verbose=verbose,
            )  # 1 to show progress bar

        elif gpus == 1:
            self.model.compile(loss="categorical_crossentropy", optimizer=opt)
            self.model.fit_generator(
                self.train_gen,
                steps_per_epoch=self.steps_per_epoch / mini_epochs,
                epochs=mini_epochs * self.epochs,
                validation_data=self.valid_gen,
                validation_steps=self.validation_steps / mini_epochs,
                callbacks=callbacks,
                max_queue_size=10,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                verbose=verbose,
            )  # 1 to show progress bar

        # Build sample_model out of the trained batch_model
        self.__build_sample_model(batch_input_length=1)  # Single-output model
        self.__build_sample_model(
            batch_input_length=self.batch_size
        )  # Multi-output model

    
    # @timed
    def predict(self, temp=1, rng_seed=None):
        """
        Generate a single SMILES string.
        
        The states of the RNN are set based on the latent input.
        
        Careful, "latent" must be: the output of self.transform()
                                   or
                                   an array of molecular descriptors.
        
        If temp>0, multinomial sampling is used instead of selecting 
        the single most probable character at each step.
        
        If temp==1, multinomial sampling without temperature scaling is used.
        
        Returns:
            A single SMILES string and its NLL.
        """
        
        # Pass rng_seed for repeatable sampling 
        if rng_seed is not None:
            np.random.seed(rng_seed)
        # Reset the states between predictions because RNN is stateful!
        self.sample_model.reset_states()

        # Prepare the input char
        startidx = self.smilesvec1._char_to_int[self.smilesvec1.startchar]
        samplevec = np.zeros((1, 1, self.smilesvec1.dims[-1]))
        samplevec[0, 0, startidx] = 1
        smiles = ""
        # Initialize Negative Log-Likelihood (NLL)
        NLL = 0
        # Loop and predict next char
        for i in range(1000):
            o = self.sample_model.predict(samplevec)
            # Multinomial sampling with temperature scaling
            if temp:
                temp = abs(temp)  # Handle negative values
                nextCharProbs = np.log(o) / temp
                nextCharProbs = np.exp(nextCharProbs)
                nextCharProbs = (
                    nextCharProbs / nextCharProbs.sum() - 1e-8
                )  # Re-normalize for float64 to make exactly 1.0 for np.random.multinomial
                sampleidx = np.random.multinomial(
                    1, nextCharProbs.squeeze(), 1
                ).argmax()

            # Else, select the most probable character
            else:
                sampleidx = np.argmax(o)

            samplechar = self.smilesvec1._int_to_char[sampleidx]
            if samplechar != self.smilesvec1.endchar:
                # Append the new character
                smiles += samplechar
                samplevec = np.zeros((1, 1, self.smilesvec1.dims[-1]))
                samplevec[0, 0, sampleidx] = 1
                # Calculate negative log likelihood for the selected character given the sequence so far
                NLL -= np.log(o[0][0][sampleidx])
            else:
                return smiles, NLL

    # @timed
    def predict_batch(self, temp=1, rng_seed=None):
        """
        Generate multiple random SMILES strings.
        
        If temp>0, multinomial sampling is used instead of selecting 
        the single most probable character at each step.
        
        If temp==1, multinomial sampling without temperature scaling is used.
        
        Low temp leads to elimination of characters with low conditional probabilities.
        """
        
        # Pass rng_seed for repeatable sampling 
        if rng_seed is not None:
            np.random.seed(rng_seed)    
        # Reset the states between predictions because RNN is stateful!
        self.multi_sample_model.reset_states()

        # Index of input char "^"
        startidx = self.smilesvec1._char_to_int[self.smilesvec1.startchar]
        # Vectorize the input char for all SMILES
        samplevec = np.zeros((self.batch_input_length, 1, self.smilesvec1.dims[-1]))
        samplevec[:, 0, startidx] = 1
        # Initialize arrays to store SMILES, their NLLs and their status
        smiles = np.array([""] * self.batch_input_length, dtype=object)
        NLL = np.zeros((self.batch_input_length,))
        finished = np.array([False] * self.batch_input_length)

        # Loop and predict next char
        for i in range(1000):
            o = self.multi_sample_model.predict(
                samplevec, batch_size=self.batch_input_length
            ).squeeze()

            # Multinomial sampling with temperature scaling
            if temp:
                temp = abs(temp)  # No negative values
                nextCharProbs = np.log(o) / temp
                nextCharProbs = np.exp(nextCharProbs)  # .squeeze()

                # Normalize probabilities
                nextCharProbs = (nextCharProbs.T / nextCharProbs.sum(axis=1) - 1e-8).T
                sampleidc = np.asarray(
                    [
                        np.random.multinomial(1, nextCharProb, 1).argmax()
                        for nextCharProb in nextCharProbs
                    ]
                )

            else:
                sampleidc = np.argmax(o, axis=1)

            samplechars = [self.smilesvec1._int_to_char[idx] for idx in sampleidc]

            for idx, samplechar in enumerate(samplechars):
                if not finished[idx]:
                    if samplechar != self.smilesvec1.endchar:
                        # Append the SMILES with the next character
                        smiles[idx] += self.smilesvec1._int_to_char[sampleidc[idx]]
                        samplevec = np.zeros(
                            (self.batch_input_length, 1, self.smilesvec1.dims[-1])
                        )
                        # One-Hot Encode the character
                        # samplevec[:,0,sampleidc] = 1
                        for count, sampleidx in enumerate(sampleidc):
                            samplevec[count, 0, sampleidx] = 1
                        # Calculate negative log likelihood for the selected character given the sequence so far
                        NLL[idx] -= np.log(o[idx][sampleidc[idx]])
                    else:
                        finished[idx] = True
                        # print("SMILES has finished at %i" %i)

            # If all SMILES are finished, i.e. the endchar "$" has been generated, stop the generation
            if finished.sum() == len(finished):
                return smiles, NLL

    @timed
    def get_smiles_nll(self, smiles_ref) -> float:
        """
        Calculate the NLL of a given SMILES string if its descriptors are used as RNN states.
        
        Returns:
            The NLL of sampling a given SMILES string.
        """

        # Reset the states between predictions because RNN is stateful!
        self.sample_model.reset_states()

        # Prepare the input char
        startidx = self.smilesvec1._char_to_int[self.smilesvec1.startchar]
        samplevec = np.zeros((1, 1, self.smilesvec1.dims[-1]))
        samplevec[0, 0, startidx] = 1

        # Initialize Negative Log-Likelihood (NLL)
        NLL = 0
        # Loop and predict next char
        for i in range(1000):
            o = self.sample_model.predict(samplevec)

            samplechar = smiles_ref[i]
            sampleidx = self.smilesvec1._char_to_int[samplechar]

            if i != len(smiles_ref) - 1:
                samplevec = np.zeros((1, 1, self.smilesvec1.dims[-1]))
                samplevec[0, 0, sampleidx] = 1
                # Calculate negative log likelihood for the selected character given the sequence so far
                NLL -= np.log(o[0][0][sampleidx])
            else:
                return NLL

    @timed
    def get_smiles_nll_batch(self, smiles_ref) -> list:
        """
        Calculate the individual NLL for a batch of known SMILES strings.
        Batch size is equal to self.batch_input_length so reset it if needed.
        
        Returns:
            NLL of sampling all listed SMILES.
        """

        # Reset the states between predictions because RNN is stateful!
        self.multi_sample_model.reset_states()

        # Index of input char "^"
        startidx = self.smilesvec1._char_to_int[self.smilesvec1.startchar]
        # Vectorize the input char for all SMILES
        samplevec = np.zeros((self.batch_input_length, 1, self.smilesvec1.dims[-1]))
        samplevec[:, 0, startidx] = 1
        # Initialize arrays to store NLLs and flag if a SMILES is finished
        NLL = np.zeros((self.batch_input_length,))
        finished = np.array([False] * self.batch_input_length)

        # Loop and predict next char
        for i in range(1000):
            o = self.multi_sample_model.predict(
                samplevec, batch_size=self.batch_input_length
            ).squeeze()
            samplechars = []

            for smiles in smiles_ref:
                try:
                    samplechars.append(smiles[i])
                except:
                    # This is a finished SMILES, so "i" exceeds dimensions
                    samplechars.append("$")

            sampleidc = np.asarray(
                [self.smilesvec1._char_to_int[char] for char in samplechars]
            )

            for idx, samplechar in enumerate(samplechars):
                if not finished[idx]:
                    if i != len(smiles_ref[idx]) - 1:
                        samplevec = np.zeros(
                            (self.batch_input_length, 1, self.smilesvec1.dims[-1])
                        )
                        # One-Hot Encode the character
                        for count, sampleidx in enumerate(sampleidc):
                            samplevec[count, 0, sampleidx] = 1
                        # Calculate negative log likelihood for the selected character given the sequence so far
                        NLL[idx] -= np.log(o[idx][sampleidc[idx]])
                    else:
                        finished[idx] = True

            # If all SMILES are finished, i.e. the endchar "$" has been generated, stop the generation
            if finished.sum() == len(finished):
                return NLL

    def summary(self):
        """
        Echo the training configuration for inspection.
        """

        print(
            "\nModel trained with dataset %s that has maxlen=%d and charset=%s for %d epochs."
            % (self.dataset_name, self.maxlen, self.charset, self.epochs)
        )

        print(
            "noise_std: %.6f, lstm_dim: %d, dec_layers: %d, td_dense_dim: %d, batch_size: %d, codelayer_dim: %d, lr: %.6f."
            % (
                self.noise_std,
                self.lstm_dim,
                self.dec_layers,
                self.td_dense_dim,
                self.batch_size,
                self.codelayer_dim,
                self.lr,
            )
        )

    def get_graphs(self):
        """
        Export the graphs of the model and its submodels to png files.
        Requires "pydot" and "graphviz" to be installed (pip install graphviz && pip install pydot).
        """

        try:
            from keras.utils import plot_model
            from keras.utils.vis_utils import model_to_dot

            # from IPython.display import SVG

            plot_model(self.model, to_file="model.png")

            print("Model exported to png.")

        except:
            print("Check pydot and graphviz installation.")

    @timed
    def save(self, model_name):
        """
        Save model in a zip file.
        """

        with tempfile.TemporaryDirectory() as dirpath:

            # Save the Keras model
            self.model.save(dirpath + "/model.h5")

            # Exclude unpicklable and unwanted attributes
            excl_attr = [
                "_DDC__mode",  # excluded because it is always identified within self.__init__()
                "_DDC__train_gen",  # unpicklable
                "_DDC__valid_gen",  # unpicklable
                "_DDC__sample_model",  # unpicklable
                "_DDC__multi_sample_model",  # unpicklable
                "_DDC__model",
            ]  # unpicklable

            # Cannot deepcopy self.__dict__ because of Keras' thread lock so this is
            # bypassed by popping and re-inserting the unpicklable attributes
            to_add = {}
            # Remove unpicklable attributes
            for attr in excl_attr:
                to_add[attr] = self.__dict__.pop(attr, None)

            # Pickle metadata, i.e. almost everything but the Keras models and generators
            pickle.dump(self.__dict__, open(dirpath + "/metadata.pickle", "wb"))

            # Zip directory with its contents
            shutil.make_archive(model_name, "zip", dirpath)

            # Finally, re-load the popped elements for the model to be usable
            for attr in excl_attr:
                self.__dict__[attr] = to_add[attr]

            print("Model saved.")
