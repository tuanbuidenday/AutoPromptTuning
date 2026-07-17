from optimization_pipeline import OptimizationPipeline
from utils.config import load_yaml, override_config
import argparse

# General Training Parameters
parser = argparse.ArgumentParser()

parser.add_argument('--basic_config_path', default='config/config_default.yml', type=str, help='Configuration file path')
parser.add_argument('--batch_config_path', default='',
                    type=str, help='Batch classification configuration file path')
parser.add_argument('--prompt',
                    default='',
                    required=False, type=str, help='Prompt to use as initial.')
parser.add_argument('--task_description',
                    default='',
                    required=False, type=str, help='Describing the task')
parser.add_argument('--load_path', default='', required=False, type=str, help='In case of loading from checkpoint')
parser.add_argument('--output_dump', default='dump', required=False, type=str, help='Output to save checkpoints')
parser.add_argument('--num_steps', default=40, type=int, help='Number of iterations')

opt = parser.parse_args()

if opt.batch_config_path == '':
    # load the basic configuration using load_yaml
    config_params = load_yaml(opt.basic_config_path)
else:
    # override the basic configuration with the batch configuration
    config_params = override_config(opt.batch_config_path, config_file=opt.basic_config_path)

if opt.task_description == '':
    task_description = input("Describe the task: ")
else:
    task_description = opt.task_description

if opt.prompt == '':
    initial_prompt = input("Initial prompt: ")
else:
    initial_prompt = opt.prompt

# For an LLM annotator with no explicit instruction, use the task description as the
# ground-truth annotation instruction so any entered prompt can be optimized without
# manual labels or an Argilla server.
if config_params.annotator.method == 'llm' and not config_params.annotator.config.get('instruction'):
    labels = config_params.dataset.label_schema
    config_params.annotator.config.instruction = (
        f"{task_description} Answer with exactly one of the following labels: "
        f"{', '.join(labels)}."
    )

# Initializing the pipeline
pipeline = OptimizationPipeline(config_params, task_description, initial_prompt, output_path=opt.output_dump)
if (opt.load_path != ''):
    pipeline.load_state(opt.load_path)
best_prompt = pipeline.run_pipeline(opt.num_steps)
print('\033[92m' + 'Calibrated prompt score:', str(best_prompt['score']) + '\033[0m')
print('\033[92m' + 'Calibrated prompt:', best_prompt['prompt'] + '\033[0m')

