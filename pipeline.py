from leave_one_out_whole_data import leave_one_out_whole_data
from calculate_shared_SSMD import calculate_shared_SSMD
from model_after_LOO import model_after_LOO 
from test_new_data_on_final_model import test_new_data_on_final_model 
from heatmap_for_xgboost import heatmap_for_xgboost
from one_model_run import one_model_run

class Config:
    def __init__(self):
        self.df_sick_path = None
        self.df_healthy_path = None
        self.block_size = None
        self.heatmap_data_input = None
        self.output_dir = None
        self.log_level = None
        self.param_grid = {}
        self.n_splits = None
        self.n_jobs = None
        self.model_path = None
        self.SSMD_data = None
        self.df_test_sick_path = None
        self.df_test_healthy_path = None
        self.precent_shared = None
        self.two_groups_for_test = None

    def fromJSON(self, json_path):
        import json
        with open(json_path, 'r') as f:
            config_data = json.load(f)

        self.df_sick_path = config_data['df_sick_path']
        self.df_healthy_path = config_data['df_healthy_path']
        self.block_size = config_data['block_size']
        self.thres_prec = config_data['thres_prec']
        self.output_dir = config_data['output_dir']
        self.log_level = config_data['log_level']
        self.param_grid = config_data.get('param_grid', {})
        self.n_splits = config_data.get('n_splits')
        self.n_jobs = config_data.get('n_jobs')
        self.heatmap_data_input = config_data.get('heatmap_data_input')
        self.model_path = config_data['model_path']
        self.SSMD_data = config_data['SSMD_data']
        self.df_test_sick_path = config_data['df_test_sick_path']
        self.df_test_healthy_path = config_data['df_test_healthy_path']
        self.precent_shared = config_data['precent_shared']
        self.two_groups_for_test = config_data['two_groups_for_test']

class Output:
    def __init__(self, output_dir, log_level):
        self.output_dir = output_dir
        self.log_level = log_level

    def WriteAndCloseOutput(self):
        print("Output written and closed.")

class Pipeline:
    def __init__(self, config, output):
        self.config = config
        self.output = output

    def leave_one_out_whole_data(self):
        print("Running leave_one_out_whole_data...")
        leave_one_out_whole_data(self.config)

    def calculate_shared_SSMD(self):
        print("Running calculate_shared_SSMD...")
        calculate_shared_SSMD(self.config)

    def model_after_LOO(self):
        print("Running model_after_LOO...")
        model_after_LOO(self.config)
        
    def heatmap_for_xgboost(self):
        print("Runing heatmap_for_xgboost...")
        heatmap_for_xgboost(self.config)
        
    def test_new_data_on_final_model(self):
        print("Running test_new_data_on_final_model...")
        test_new_data_on_final_model(self.config)
        
    def one_model_run(self):
        print("Running one_model_run...")
        one_model_run(self.config)


