# Configuration class for TimeMixer
class TimeMixerConfig_imputation:
    def __init__(self):
        # Task configuration
        self.task_name = 'imputation'
        
        # Model dimensions - based on checkpoint name parameters
        self.seq_len = 30        # sl30
        self.label_len = 10      # ll10
        self.pred_len = 10       # pl10
        self.enc_in = 46         # number of input features
        self.c_out = 46          # number of output features
        self.d_model = 32        # dm32
        self.d_ff = 32           # df32
        
        # Transformer configuration
        self.n_heads = 8         # nh8
        self.e_layers = 4        # el4
        self.d_layers = 1        # dl1
        self.factor = 3          # fc3
        self.dropout = 0.1
        
        # Embedding configuration
        self.embed = 'timeF'     # ebtimeF
        self.freq = 'h'
        
        # TimeMixer specific configuration
        self.down_sampling_window = 2
        self.down_sampling_layers = 1    # Reduced from 2 to 1 based on checkpoint shapes
        self.down_sampling_method = 'avg'
        self.channel_independence = False  # dtTrue means use_norm=True, but channel_independence should be False for proper weight loading
        
        # Decomposition configuration
        self.decomp_method = 'moving_avg'
        self.moving_avg = 25
        self.top_k = 5
        
        # Normalization
        self.use_norm = 1
        
        # Other configurations
        self.expand = 2          # expand2
        self.d_conv = 4          # dc4
        
        # For classification tasks
        self.num_class = 2

class TimeMixerConfig_forecast:
    def __init__(self):
        # Task configuration
        self.task_name = 'short_term_forecast'
        self.seasonal_patterns = 'Minutely'
        self.model_id = 'm4_Monthly'
        self.model = 'TimeMixer'
        self.features = 'M'
        
        # Model dimensions - based on checkpoint name parameters
        self.seq_len = 200        
        self.enc_in = 46         # number of input features
        self.dec_in = 8
        self.c_out = 46          # number of output features
        self.d_model = 32        # dm32
        self.d_ff = 32           # df32
        
        # Transformer configuration
        self.n_heads = 8         # nh8
        self.e_layers = 4        # el4
        self.d_layers = 1        # dl1
        self.factor = 3          # fc3
        self.dropout = 0.1
        
        # Embedding configuration
        self.embed = 'timeF'     # ebtimeF
        self.freq = 'h'
        
        # TimeMixer specific configuration
        self.down_sampling_window = 2
        self.down_sampling_layers = 1    # Reduced from 2 to 1 based on checkpoint shapes
        self.down_sampling_method = 'avg'
        self.channel_independence = True  # dtTrue means use_norm=True, but channel_independence should be False for proper weight loading
        
        # Decomposition configuration
        self.decomp_method = 'moving_avg'
        self.moving_avg = 25
        self.top_k = 5
        
        # Normalization
        self.use_norm = 1
        
        # Other configurations
        self.expand = 2          # expand2
        self.d_conv = 4          # dc4
        
        # For classification tasks
        self.num_class = 2

        self.label_len = 48
        self.pred_len = 96
        

class TimesNetConfig:
    def __init__(self):
        self.task_name = 'anomaly_detection'
        
        self.model_id = 'm4_Monthly'
        self.model = 'TimesNet'
        self.seq_len = 200
        self.enc_in = 46
        self.dec_in = 8
        self.c_out = 46
        self.d_model = 8
        self.d_ff = 16
        self.n_heads = 8
        self.e_layers = 1
        self.top_k = 3

        # Model dimensions - based on checkpoint name parameters
        self.label_len = 10      # ll10
        self.pred_len = 0       # pl10

        
        # Transformer configuration
        self.d_layers = 1        # dl1
        self.factor = 3          # fc3
        self.dropout = 0.1
        
        # Embedding configuration
        self.embed = 'timeF'     # ebtimeF
        self.freq = 'h'
        
        # TimeMixer specific configuration
        self.down_sampling_window = 2
        self.down_sampling_layers = 1    # Reduced from 2 to 1 based on checkpoint shapes
        self.down_sampling_method = 'avg'
        self.channel_independence = False  # dtTrue means use_norm=True, but channel_independence should be False for proper weight loading
        
        # Decomposition configuration
        self.decomp_method = 'moving_avg'
        self.moving_avg = 25
        self.top_k = 5
        
        # Normalization
        self.use_norm = 1
        
        # Other configurations
        self.expand = 2          # expand2
        self.d_conv = 4          # dc4
        

        self.num_kernels = 6