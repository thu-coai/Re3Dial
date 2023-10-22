import os
from multiprocessing import Pool
import argparse


def run_cmd(cmd):
    os.system(cmd)


def run_exp(input_info_list):
    cmds = []

    for item in input_info_list:
        query_file, source_file, output_file, gpu_id = item
        cmd = f"bash scripts/retrieve.sh {source_file} {query_file} {output_file} {gpu_id}"
        cmds.append(cmd)

    pool = Pool(len(cmds))
    for cmd in cmds:
        pool.apply_async(run_cmd, args=(cmd,))

    print("start searching")
    pool.close() # close pool, no more new task
    pool.join() # main process wait for subprocess to exit
    print('finish searching')


def main(prefix, output_prefix, start_idx, end_idx, num_gpus):
    file_idx = start_idx
    input_info = []
    while file_idx <= end_idx:
        str_index = "%02d" % file_idx
        query_file = f'{prefix}_{str_index}_embedding_query.dataset'
        source_file = f'{prefix}_{str_index}_embedding_context.dataset'
        output_file = f'{output_prefix}/{str_index}'
    
        cur_gpu_id = len(input_info)
        input_info.append(
            (
                query_file, 
                source_file,
                output_file,
                cur_gpu_id
            )
        )
        
        if len(input_info) == num_gpus or file_idx == end_idx:
            print('input_info = ', input_info)
            run_exp(input_info)
            input_info = [] # 重置

        file_idx += 1
            

if __name__ == "__main__":
    in_dir = ''
    out_dir = ''
    
    start_idx = 0
    end_idx = 20
    num_gpus = 8
    main(in_dir, out_dir, start_idx, end_idx, num_gpus)