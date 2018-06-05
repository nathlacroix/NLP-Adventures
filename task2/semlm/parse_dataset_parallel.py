import argparse
from time import sleep
from pathlib import Path

from multiprocessing.dummy import Pool
from subprocess import Popen, STDOUT

temporary_folder = './tmp/'
num_chunks = 5
start_delay_s = 120

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help="path to the stories")
    parser.add_argument('output_path', type=str, help="path to the output file")
    parser.add_argument('--has_endings', dest='has_endings', action='store_true')
    parser.add_argument('--has_titles', dest='has_titles', action='store_true')
    parser.add_argument('--srl_annotation', default='framenet', type=str)
    args = parser.parse_args()

    assert Path(args.data_path).exists()
    with open(args.data_path, 'r') as f:
        header = f.readline()
        data = f.readlines()

    chunk_size = len(data) // num_chunks + 1
    Path(temporary_folder).mkdir()
    input_files = [Path(temporary_folder, str(i)) for i in range(num_chunks)]
    output_files = [Path(temporary_folder, str(i)+'_out') for i in range(num_chunks)]

    # Write temporary input files
    for i, t in enumerate(input_files):
        with open(t, 'w') as f:
            f.write(header)
            chunk = data[i*chunk_size:(i+1)*chunk_size]
            f.writelines(chunk)

    def get_subprocess_args(i):
        parse_script = Path(Path(__file__).parents[0], 'parse_dataset.py').as_posix()
        sub_args = ['python3', parse_script,
                    input_files[i].as_posix(), output_files[i].as_posix(),
                    '--srl_annotation={}'.format(args.srl_annotation)]
        if args.has_endings:
            sub_args.append('--has_endings')
        if args.has_titles:
            sub_args.append('--has_titles')
        return sub_args

    # Spawn subprocesses
    processes = []
    for i in range(num_chunks):
        processes.append(Popen(get_subprocess_args(i), shell=False,
                               stderr=STDOUT))
        sleep(start_delay_s)

    # Wait for termination
    exit_codes = Pool(len(processes)).map(lambda p: p.wait(), processes)
    for i, c in enumerate(exit_codes):
        assert not c, 'Parsing script #{} exited with non-zero code: {}'.format(
                i, exit_codes)

    # Gather outputs and aggregate
    output = []
    for o in output_files:
        with open(o, 'r') as f:
            output.extend(f.readlines())
    with open(args.output_path, 'w') as f:
        f.writelines(output)
