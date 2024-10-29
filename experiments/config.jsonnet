// This is a jsonnet file for experiment config.
local UnImplementedError = function() {
  'error': error 'This is an abstract class, should be implemented',
};
// What is a experiment config?
// It is a json that contains the following fields:
local BaseConfig = {
  // data is stored at
  // data_root/${data_folder}/${task}/${to_display_model_name(model_str)}_${to_display_model_name(ref_model_str)}
  data_folder: UnImplementedError(),  // string
  // The name of the experiment.
  task: UnImplementedError(),  // string
  // model parameters
  model_str: UnImplementedError(),  // string
  ref_model_str: UnImplementedError(),  // string
  // dataset parameters
  ds_name: UnImplementedError(),  // string
  ds_cut_len: UnImplementedError(),  // int
  // worker parameters
  device: UnImplementedError(),  // string
  max_length: UnImplementedError(),  // int
  batch_size: UnImplementedError(),  // int
  private_key: UnImplementedError(),  // string
  methods: UnImplementedError(),  // list[string]
  ns: UnImplementedError(),  // list
  seeds: UnImplementedError(),  // list
  reweights: UnImplementedError(),  // list[string]
  print_output: UnImplementedError(),  // bool
  assert_cch: UnImplementedError(),  // bool
  assert_log_p_values: UnImplementedError(),  // bool
  // ray parameters
  repartition_size: UnImplementedError(),  // bool
};
local large_llamas = [
  'huggyllama/llama-7b',
  'huggyllama/llama-13b',
  'huggyllama/llama-65b',
  'daryl149/llama-2-7b-chat-hf',
  'daryl149/llama-2-13b-chat-hf',
  'daryl149/llama-2-70b-chat-hf',
];
local small_llamas = ['JackFram/llama-68m', 'JackFram/llama-160m'];

local large_opts = [
  'facebook/opt-6.7b',
  'facebook/opt-13b',
  'facebook/opt-30b',
  'facebook/opt-66b',
];
local small_opts = ['facebook/opt-125m', 'facebook/opt-350m'];

local large_gpts = [
  'openai-community/gpt2-large',
  'openai-community/gpt2-xl',
  'EleutherAI/gpt-neo-2.7B',
  'EleutherAI/gpt-neo-20B',
];
local small_gpts = ['gpt2', 'gpt2-medium'];

local DefaultConfig = BaseConfig {
  device: 'cuda:0',
  max_length: 128,
  batch_size: 4,
  private_key: '1234',
  print_output: false,
  methods: ['basic', 'basic_uwm', 'mc', 'mc_uwm_strength', 'mc_uwm_speed'],
  reweights: ['deltagumbel', 'gamma'],
  assert_cch: true,
  assert_log_p_values: true,
};
local Debug_Config = DefaultConfig {
  data_folder: 'debug_data',
  task: 'summarization_scan_n',
  model_str: large_llamas[0],
  ref_model_str: small_llamas[0],
  ds_name: 'summarization',
  ds_cut_len: 1,
  ns: [1],
  seeds: std.range(1, 3),
  // reweights: ['deltagumbel'],
  batch_size: 10000,
  repartition_size: 1,
  // methods: ['mc', 'mc_uwm_strength'],
  // methods: ['mc', 'mc_uwm_speed'],
  methods: ['mc', 'basic'],
  assert_cch: true,
  assert_log_p_values: true,
  print_output: true,
};
local Exp1_Verify_Config = DefaultConfig {
  data_folder: 'verify_data',
  task: 'summarization_scan_n',
  model_str: large_llamas[0],
  ref_model_str: small_llamas[0],
  ds_name: 'summarization',
  ds_cut_len: 10,
  ns: [1],
  seeds: std.range(1, 10),
  repartition_size: 500,
};
local Exp2_Config_Template = DefaultConfig {
  data_folder: 'data',
  task: 'summarization_scan_n',
  // model_str: large_llamas[0],
  ref_model_str: small_llamas[0],
  ds_name: 'summarization',
  // ds_cut_len: 10,
  ds_cut_len: 100,
  // ns: [1, 2],
  ns: [1, 2, 3, 4],
  // methods: ['mc', 'mc_uwm_strength', 'mc_uwm_speed'],
  seeds: std.range(1, 100),
  repartition_size: 5000,
};
local Exp3_Config_Template = Exp2_Config_Template {
  data_folder: 'data',
  task: 'oeg_scan_n',
  ds_name: 'oeg',
};
local configs = [
                  // Debug_Config,
                  // Exp1_Verify_Config,
                ]
                + [
                  Exp2_Config_Template {
                    model_str: model_str,
                  }
                  for model_str in large_llamas[:2]
                ]
                + [
                  Exp3_Config_Template {
                    model_str: model_str,
                  }
                  for model_str in large_llamas[:2]
                ]
;
configs
