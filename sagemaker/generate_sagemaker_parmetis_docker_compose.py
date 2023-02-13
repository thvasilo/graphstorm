from typing import Dict
import argparse
import sys
import yaml

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-instances", required=True, type=int)
    parser.add_argument("--graph-name", required=True)
    parser.add_argument("--graph-data-s3", required=True)
    parser.add_argument("--output-data-s3", required=True)
    parser.add_argument("--num-parts", required=True, type=int)
    parser.add_argument("--region", required=True)
    parser.add_argument("--image", required=False, default='graphstorm-dev:v1-0.0.0',
        help="Local image name and tag that we will use for the containers.")

    return parser.parse_args(args)

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    compose_dict = dict()

    compose_dict['version'] = '3.7'
    compose_dict['networks'] = {'mpi': {'name': 'mpi-network'}}

    def generate_instance_entry(instance_idx: int, world_size: int) -> Dict[str, str]:
        inner_host_list = [f'algo-{i}' for i in range(1, world_size+1)]
        quoted_host_list = ', '.join(f'"{host}"' for host in inner_host_list)
        host_list = f'[{quoted_host_list}]'
        return {
                'image': args.image,
                'container_name': f'algo-{instance_idx}',
                'hostname': f'algo-{instance_idx}',
                'networks': ['mpi'],
                'command': f'python3 sagemaker_parmetis.py --graph-name "{args.graph_name}" --graph-data-s3 "{args.graph_data_s3}" --num-parts {args.num_parts} --output-data-s3 "{args.output_data_s3}"',
                'environment':
                    {'SM_TRAINING_ENV': f'{{"hosts": {host_list}, "current_host": "algo-{instance_idx}"}}',
                    'WORLD_SIZE': world_size,
                    'MASTER_ADDR': 'algo-1',
                    'AWS_REGION': args.region},
                'ports': [22],
                'working_dir': '/opt/ml/code/'
            }

    service_dicts = {f"algo-{i}": generate_instance_entry(i, args.num_instances) for i in range(1, args.num_instances+1)}

    compose_dict['services'] = service_dicts

    with open(f'docker-compose-{args.graph_name}-{args.num_instances}workers-{args.num_parts}parts.yml', 'w') as f:
        yaml.dump(compose_dict, f)
